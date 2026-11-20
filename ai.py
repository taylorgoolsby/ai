#!/usr/bin/env python3
"""ai - a local agentic coding assistant for your terminal (LM Studio).

Runs fine when piped in:  curl -s <raw url>/ai.py | python3 - "$@"
First run installs its two dependencies (textual, httpx) into ~/.cache/ai/deps.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------
# Bootstrap: make sure textual + httpx are importable without touching the
# system Python (pip --target into a private cache dir).
# --------------------------------------------------------------------------
try:
    import fcntl  # POSIX: lets a session log be locked against a second ai process
except ImportError:
    fcntl = None

HOME = Path(os.getenv("AI_HOME") or Path.home() / ".cache" / "ai")
DEPS = HOME / "deps"


def _bootstrap() -> None:
    sys.path.append(str(DEPS))
    try:
        import httpx  # noqa: F401
        import textual  # noqa: F401
        return
    except ImportError:
        pass
    print("ai: installing dependencies (first run only)...", file=sys.stderr)
    DEPS.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--target", str(DEPS), "textual", "httpx"]
    )
    if r.returncode != 0:
        sys.exit("ai: could not install dependencies. Try: python3 -m pip install textual httpx")
    importlib.invalidate_caches()


_bootstrap()

import httpx  # noqa: E402
from rich.text import Text  # noqa: E402
from textual import work  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Horizontal, Vertical, VerticalScroll  # noqa: E402
from textual.screen import ModalScreen  # noqa: E402
from textual.widgets import Button, Footer, Header, Input, Static  # noqa: E402
from textual.widgets import Markdown as MarkdownView  # noqa: E402


# --------------------------------------------------------------------------
# LM Studio client (OpenAI-compatible dev server)
# --------------------------------------------------------------------------
class LLMError(Exception):
    pass


ELLIPSIS = "\u2026"
DEFAULT_TIMEOUT = 600.0             # longest we wait for LM Studio to send *any* data
DEFAULT_GENERATION_TIMEOUT = 600.0  # longest a single reply may run before it is stopped (breaks output loops)


class GenerationStopped(LLMError):
    """The user pressed stop."""

    def __init__(self, message: str = "Stopped."):
        super().__init__(message)


class GenerationTimeout(LLMError):
    """One reply ran longer than the generation limit (typically a model stuck repeating itself)."""

    def __init__(self, seconds: float):
        super().__init__(f"The model was still generating after {format_elapsed(seconds)}; it was stopped.")
        self.seconds = seconds


def _stream_detail(name: str, args_text: str) -> str:
    """Live detail for a tool call still being generated: its target and how much has been written."""
    m = re.search(r'"(?:path|command|pattern)"\s*:\s*"((?:[^"\\]|\\.)*)', args_text)
    target = ""
    if m:
        target = m.group(1)
        try:
            target = json.loads('"' + target + '"')  # un-escape; may fail on a half-written escape
        except ValueError:
            pass
        target = _one_line(target, 60)
    size = f"{len(args_text):,} chars"
    return f"{target} \u00b7 {size}" if target else size


class LMStudio:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT, generation_timeout: float | None = DEFAULT_GENERATION_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Replies are streamed, so `timeout` is the longest we wait for *any* data, not for the whole
        # reply. `generation_timeout` is the longest a whole reply may take (0/None = no limit): it
        # stops a model that keeps producing tokens forever, e.g. stuck repeating itself.
        self.timeout = timeout
        self.generation_timeout = generation_timeout or 0
        self._lock = threading.Lock()
        self._active = False          # a chat() call is in flight
        self._response = None         # its streaming response, once headers arrived
        self._abort_reason: str | None = None
        self._stale_client = False    # the client was closed to abort a request and must be rebuilt
        self.http = self._new_client()

    def _new_client(self) -> httpx.Client:
        return httpx.Client(timeout=httpx.Timeout(self.timeout, connect=5.0))

    def _ensure_client(self) -> None:
        if self._stale_client:
            self.http, self._stale_client = self._new_client(), False

    def _translate(self, e: httpx.HTTPError) -> LLMError:
        if isinstance(e, httpx.ConnectError):
            return LLMError(
                f"Can't reach LM Studio at {self.base_url}. "
                "Load a model and start the server (Developer tab -> Start Server, "
                "or `lms server start`)."
            )
        if isinstance(e, httpx.ReadTimeout):
            return LLMError(
                f"LM Studio sent no data for {self.timeout:g}s. It may still be processing a very long prompt "
                "or be stuck; raise --timeout (or AI_TIMEOUT), shrink the conversation with /clear, "
                "or use a faster model."
            )
        if isinstance(e, httpx.HTTPStatusError):
            return LLMError(f"LM Studio returned {e.response.status_code}: {e.response.text[:500]}")
        return LLMError(str(e) or type(e).__name__)

    def _request(self, method: str, path: str, **kw) -> dict:
        self._ensure_client()
        try:
            r = self.http.request(method, self.base_url + path, **kw)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise self._translate(e)

    def list_models(self) -> list[str]:
        return [m["id"] for m in self._request("GET", "/models").get("data", [])]

    def ensure_model(self) -> str:
        if not self.model:
            models = [m for m in self.list_models() if "embed" not in m.lower()]
            if not models:
                raise LLMError("No model is loaded in LM Studio.")
            self.model = models[0]
        return self.model

    def _shutdown_connections(self) -> None:
        """Make a blocked read return *now*. Closing a socket from another thread doesn't wake a thread
        that is blocked reading it; shutting it down does (and tells LM Studio to stop generating).
        Reaches into httpcore's pool, so it quietly does nothing if those internals ever change."""
        try:
            for conn in self.http._transport._pool.connections:
                sock = conn._connection._network_stream.get_extra_info("socket")
                if sock is not None:
                    sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass

    def abort(self, reason: str = "stopped") -> None:
        """Abort the reply being generated (safe to call from any thread; a no-op when idle).
        Closing the connection is what makes LM Studio stop generating."""
        with self._lock:
            if not self._active:
                return
            self._abort_reason = self._abort_reason or reason
            response = self._response
        self._shutdown_connections()
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        else:  # still connecting / waiting for the first byte: tear the connection pool down
            self._stale_client = True
            try:
                self.http.close()
            except Exception:
                pass

    def _aborted(self, reason: str) -> LLMError:
        return GenerationTimeout(self.generation_timeout) if reason == "timeout" else GenerationStopped()

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             progress: Callable[[str, str], None] | None = None,
             on_text: Callable[[str, str], None] | None = None,
             cancelled: Callable[[], bool] | None = None) -> dict:
        """Send the whole conversation and stream the reply. Returns the assistant message
        ({"content", optional "tool_calls", "reasoning_content", "finish_reason"}).

        progress(label, detail): throttled status while the reply is generated.
        on_text(channel, text_so_far): throttled live text; channel is "reasoning", "answer" or "tool".
        cancelled(): polled; when true the request is dropped and GenerationStopped is raised.
        Raises GenerationTimeout if the reply takes longer than generation_timeout."""
        body = {"model": self.ensure_model(), "messages": messages, "temperature": 0.2, "stream": True}
        if tools:
            body["tools"] = tools
        with self._lock:
            self._ensure_client()
            self._active, self._response, self._abort_reason = True, None, None
        watchdog = None
        if self.generation_timeout:
            watchdog = threading.Timer(self.generation_timeout, self.abort, args=("timeout",))
            watchdog.daemon = True
            watchdog.start()

        def check() -> None:
            reason = self._abort_reason or ("stopped" if cancelled and cancelled() else None)
            if reason:
                raise self._aborted(reason)

        try:
            check()
            with self.http.stream("POST", self.base_url + "/chat/completions", json=body) as r:
                with self._lock:
                    self._response = r
                check()
                if r.status_code >= 400:
                    r.read()
                    raise LLMError(f"LM Studio returned {r.status_code}: {r.text[:500]}")
                ctype = r.headers.get("content-type", "")
                if "json" in ctype and "event-stream" not in ctype:  # server ignored stream=true
                    r.read()
                    try:
                        data = r.json()
                    except ValueError:
                        raise LLMError(f"Unexpected response: {r.text[:500]}")
                    try:
                        return data["choices"][0]["message"]
                    except (KeyError, IndexError, TypeError):
                        raise LLMError(f"Unexpected response: {str(data)[:500]}")
                msg = self._consume(r.iter_lines(), progress, on_text, check)
                with self._lock:
                    self._active = False  # finished: a late watchdog must not turn this into a timeout
                return msg
        except Exception as e:
            if self._abort_reason:  # closing a stream under a blocked read surfaces as assorted errors
                raise self._aborted(self._abort_reason) from None
            if isinstance(e, httpx.HTTPError):
                raise self._translate(e) from None
            raise
        finally:
            if watchdog:
                watchdog.cancel()
            with self._lock:
                self._active, self._response = False, None

    @staticmethod
    def _consume(lines, progress, on_text=None, aborted=None) -> dict:
        """Assemble a message from server-sent-event lines (OpenAI streaming format).
        `aborted()` is called per line and raises to abandon the stream."""
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict] = {}
        finish = None
        state = {"label": "", "t": 0.0}
        text_state = {"t": float("-inf")}
        sent = {"reasoning": 0, "answer": 0, "tool": 0}

        def report(label: str, detail: str, now: float) -> None:
            if not progress:
                return
            if label != state["label"] or now - state["t"] >= 0.15:  # label changes always get through
                state["label"], state["t"] = label, now
                progress(label, detail)

        def flush(now: float, force: bool = False) -> None:
            """Push the text generated so far to the UI, at most ~10 times a second."""
            if not on_text or (not force and now - text_state["t"] < 0.1):
                return
            text_state["t"] = now
            texts = {
                "reasoning": "".join(reasoning),
                "answer": "".join(content),
                "tool": "\n".join(f"{c['name'] or ELLIPSIS} {c['args']}" for _, c in sorted(calls.items())),
            }
            for channel, text in texts.items():
                if len(text) != sent[channel]:
                    sent[channel] = len(text)
                    on_text(channel, text)

        last_idx = -1
        n_content = n_reasoning = 0
        for raw in lines:
            if aborted:
                aborted()
            line = raw.strip()
            if not line.startswith("data:"):
                continue  # blank lines, ": keep-alive" comments, event:/id: fields
            now = time.monotonic()
            line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error"):
                raise LLMError(f"LM Studio error: {str(chunk['error'])[:500]}")
            for choice in chunk.get("choices") or []:
                delta = choice["delta"] if "delta" in choice else (choice.get("message") or {})
                finish = choice.get("finish_reason") or finish

                r_text = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(r_text, str) and r_text:
                    reasoning.append(r_text)
                    n_reasoning += len(r_text)
                    report("Reasoning", f"{n_reasoning:,} chars", now)

                c_text = delta.get("content")
                if isinstance(c_text, str) and c_text:
                    content.append(c_text)
                    n_content += len(c_text)
                    report("Writing a reply", f"{n_content:,} chars", now)

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index")
                    if idx is None:  # no index: a *different* id starts a new call, otherwise continue the last one
                        prev = calls.get(last_idx)
                        new_call = prev is None or bool(tc.get("id") and prev["id"] and tc["id"] != prev["id"])
                        idx = last_idx + 1 if new_call else last_idx
                    last_idx = idx
                    entry = calls.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    if name:  # some servers send the name once, some in pieces, some repeat it in every chunk
                        if not entry["name"]:
                            entry["name"] = name
                        elif name != entry["name"] and not entry["name"].endswith(name):
                            entry["name"] += name
                    args = fn.get("arguments")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    if isinstance(args, str):
                        entry["args"] += args
                    label = f"Composing {entry['name']}" if entry["name"] else "Composing a tool call"
                    report(label, _stream_detail(entry["name"], entry["args"]), now)
            flush(now)
        flush(time.monotonic(), force=True)

        msg: dict = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
        if calls:
            msg["tool_calls"] = [
                {"id": c["id"] or f"call_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                for i, c in sorted(calls.items())
            ]
        if finish:
            msg["finish_reason"] = finish
        if not (content or reasoning or calls):
            raise LLMError("LM Studio returned an empty reply.")
        return msg


# --------------------------------------------------------------------------
# Tools (sandboxed to the workspace root)
# --------------------------------------------------------------------------
MAX_OUT = 8000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def _clip(text: str, limit: int = MAX_OUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    dangerous: bool = False  # requires user approval
    validate: Callable[..., str | None] | None = None  # returns a reason to hard-block the call

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def _params(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


# --------------------------------------------------------------------------
# Shell guardrails. These block known-dangerous commands before the approval
# prompt is even shown: privilege escalation, deleting outside the workspace or
# .git (including via variables, globs and brace expansion), reading or writing
# paths outside the workspace, sending secrets files over the network, running
# downloaded files, destructive git operations, and so on. Commands also run
# with a scrubbed environment (see _clean_env).
# They are a seatbelt, not a sandbox: code run through an interpreter
# (`python3 -c ...`, or a script written with write_file) is not inspected, and
# network access is not restricted. Use a container for real isolation.
# --------------------------------------------------------------------------
ALWAYS_BLOCKED = {
    "sudo", "su", "doas", "pkexec", "mkswap", "fdisk", "sfdisk", "parted", "wipefs",
    "shutdown", "reboot", "halt", "poweroff", "init", "telinit", "mount", "umount",
    "chroot", "killall",
    "watch", "crontab",  # run command strings / schedule persistent jobs
}
DESTRUCTIVE = {"rm", "rmdir", "mv", "shred", "truncate", "unlink"}
MUTATORS = DESTRUCTIVE | {"cp", "chmod", "chown", "chgrp", "ln", "install", "rsync", "tee", "touch", "mkdir"}
SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
INTERPRETERS = SHELLS | {"python", "python3", "perl", "ruby", "node", "php", "deno", "bun", "lua"}
SOURCERS = {"source", "."}
# Commands that read files; their path arguments must stay inside the workspace,
# the same rule the read_file tool follows.
READERS = {
    "cat", "tac", "head", "tail", "less", "more", "nl", "wc", "sort", "uniq", "cut", "strings", "xxd", "od",
    "hexdump", "base64", "file", "stat", "diff", "cmp", "md5sum", "sha1sum", "sha256sum", "realpath",
    "readlink", "grep", "egrep", "fgrep", "rg", "ag", "awk", "sed", "ls", "tree", "du",
    "zip", "unzip", "gzip", "gunzip",
}
PATTERN_FIRST = {"grep", "egrep", "fgrep", "rg", "ag", "awk", "sed"}  # first positional is a pattern/script
_PATTERN_OPTS = {"-e", "-f", "--regexp", "--file", "--expression"}
_READER_SHORT_VALUES = {"grep": "efABCm", "egrep": "efABCm", "fgrep": "efABCm", "rg": "efABCmgt",
                        "ag": "ABC", "awk": "Ffv", "sed": "ef"}
_READER_LONG_VALUES = {"--regexp", "--file", "--expression"}
# Wrappers run another command given as their arguments (`timeout 5 rm -rf x`).
WRAPPERS = {"command", "builtin", "exec", "nohup", "time", "nice", "env", "stdbuf", "timeout", "setsid",
            "ionice", "taskset", "chrt", "flock", "caffeinate", "unbuffer", "busybox", "toybox"}
# Shell reserved words that can precede a command (`do rm x`, `then sudo ls`, `{ rm x; }`).
KEYWORDS = {"if", "then", "else", "elif", "do", "while", "until", "!", "{", "}"}
NETWORK_GET = {"curl", "wget"}
KNOWN = (ALWAYS_BLOCKED | MUTATORS | SHELLS | READERS | NETWORK_GET
         | {"find", "xargs", "git", "dd", "kill", "cd", "pushd", "eval", "tar", "source"})
OPERATOR_CHARS = set(";&|()`")
REDIRECTS = {">", ">>", "&>", "&>>", ">|", ">&"}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
SENSITIVE_FILE = re.compile(
    r"(^|/)(\.env(\..+)?|\.netrc|\.npmrc|\.pypirc|\.git-credentials|credentials|id_(rsa|dsa|ecdsa|ed25519)"
    r"|.+\.(pem|key|p12|pfx|jks|keystore)|secrets?(\.[a-z]+)?)$", re.I)
# git config keys that make git run arbitrary commands
_RISKY_GIT_CONFIG = re.compile(
    r"^(alias\.|core\.(sshcommand|fsmonitor|pager|editor|hookspath|askpass|gitproxy)|credential\.|filter\.|"
    r"diff\..*\.(textconv|command)|merge\..*\.driver|gpg\.program|sequence\.editor|url\..*\.insteadof)")
_HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_RUNS_STDIN = re.compile(r"\b(sh|bash|zsh|dash|ksh|fish|python[\d.]*|perl|ruby|node|php|deno|bun|lua|eval|source|xargs)\b")

_FETCH_WRITES = {  # (short flags taking a path we'd write, long flags)
    "curl": ("oDc", {"--output", "--dump-header", "--cookie-jar", "--output-dir", "--stderr", "--trace",
                     "--trace-ascii", "--etag-save"}),
    "wget": ("OPoa", {"--output-document", "--directory-prefix", "--output-file", "--append-output"}),
}
_FETCH_SENDS = {  # options whose value is a local file that would be read/uploaded
    "curl": ("TK", {"--upload-file", "--config"}),
    "wget": ("", {"--post-file", "--body-file"}),
}
_CURL_DATA = ("dF", {"--data", "--data-binary", "--data-urlencode", "--data-ascii", "--form", "--json"})


def _tokenize(command: str) -> list[str]:
    lex = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|()`<>")
    lex.whitespace_split = True
    return list(lex)


def _segments(tokens: list[str]) -> list[tuple[str, list[str]]]:
    segs: list[tuple[str, list[str]]] = []
    cur: list[str] = []
    sep = ""
    for t in tokens:
        # `<(` and `>(` (process substitution) start a nested command just like `(`
        if t and (set(t) <= OPERATOR_CHARS or (t[-1] == "(" and set(t) <= OPERATOR_CHARS | {"<", ">"})):
            if cur:
                segs.append((sep, cur))
            cur, sep = [], t
        else:
            cur.append(t)
    if cur:
        segs.append((sep, cur))
    return segs


def _command_start(words: list[str]) -> list[str]:
    i = 0
    while i < len(words) and (ASSIGN.match(words[i]) or words[i] in KEYWORDS):
        i += 1
    if i < len(words) and os.path.basename(words[i]) in WRAPPERS:
        for j in range(i + 1, len(words)):
            if os.path.basename(words[j]) in KNOWN:
                return words[j:]
        # unknown command behind a wrapper: skip the wrapper's own flags/numbers/assignments
        rest = words[i + 1:]
        k = 0
        while k < len(rest) and (rest[k].startswith("-") or ASSIGN.match(rest[k])
                                 or re.fullmatch(r"[\d.]+[smhd]?", rest[k])):
            k += 1
        return rest[k:]
    return words[i:]


def _is_session_file(p: Path) -> bool:
    return p.name.startswith(SESSION_PREFIX)


def _resolve_in_root(arg: str, root: Path) -> tuple[Path | None, str | None]:
    if arg.startswith("~") or "$" in arg or "`" in arg:
        return None, f"can't verify path '{arg}' (uses ~ or a variable); use a literal path inside the workspace"
    p = (root / arg).resolve()
    if not p.is_relative_to(root):
        return None, f"path '{arg}' is outside the workspace"
    return p, None


def _getopt(args: list[str], short_val: str = "", long_val=(), short_opt: str = "") -> tuple[list[tuple[str, str | None]], list[str]]:
    """Tiny getopt. Returns (opts, positionals); opts are (name, value-or-None).
    short_val: short flags that take a value (attached `-ofile` or next arg).
    short_opt: short flags whose value, if any, must be attached (sed -i[SUFFIX])."""
    opts: list[tuple[str, str | None]] = []
    pos: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            pos.extend(args[i + 1:])
            break
        if a.startswith("--"):
            name, eq, val = a.partition("=")
            if name in long_val and not eq:
                opts.append((name, args[i + 1] if i + 1 < len(args) else None))
                i += 1
            else:
                opts.append((name, val if eq else None))
        elif a.startswith("-") and len(a) > 1:
            j = 1
            while j < len(a):
                c = a[j]
                if c in short_val:
                    rest = a[j + 1:]
                    if rest:
                        opts.append(("-" + c, rest))
                    else:
                        opts.append(("-" + c, args[i + 1] if i + 1 < len(args) else None))
                        i += 1
                    break
                if c in short_opt:
                    opts.append(("-" + c, a[j + 1:]))
                    break
                opts.append(("-" + c, None))
                j += 1
        else:
            pos.append(a)
        i += 1
    return opts, pos


def _brace_expand(s: str, limit: int = 256) -> list[str] | None:
    """Expand {a,b} alternatives like a shell so every result can be checked.
    Returns None if that would produce more than `limit` paths."""
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        depth, commas, j = 0, [], i
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            elif s[j] == "," and depth == 1:
                commas.append(j)
            j += 1
        if j < len(s) and commas:  # a balanced group with top-level commas
            bounds = [i] + commas + [j]
            out: list[str] = []
            for k in range(len(bounds) - 1):
                alt = s[bounds[k] + 1:bounds[k + 1]]
                sub = _brace_expand(s[:i] + alt + s[j + 1:], limit)
                if sub is None:
                    return None
                out += sub
                if len(out) > limit:
                    return None
            return out
    return [s]


def _dot_glob(path: str) -> bool:
    """True for patterns like .* or .g* that the shell would expand to hidden files such as .git."""
    return any(p.startswith(".") and p not in (".", "..") and any(c in p for c in "*?[") for p in path.split("/"))


def _is_interpreter(cmd: str) -> bool:
    return cmd in INTERPRETERS or bool(re.fullmatch(r"(python|pypy|ruby|perl|node)[\d.]+", cmd))


def _check_read_path(cmd: str, a: str, root: Path) -> str | None:
    """Reads must stay inside the workspace. Unverifiable $VAR paths are let through
    (loops over files are common) except $HOME."""
    if a in ("", "-", "/dev/null"):
        return None
    if "$" in a:
        return f"{cmd}: can't read paths under $HOME" if re.search(r"\$\{?HOME\b", a) else None
    _, err = _resolve_in_root(a, root)
    return f"{cmd}: {err}" if err else None


def _check_send_path(cmd: str, ref: str, root: Path) -> str | None:
    ref = ref.strip()
    if ref in ("", "-"):
        return None
    p, err = _resolve_in_root(ref, root)
    if err:
        return f"{cmd}: {err}"
    if SENSITIVE_FILE.search(str(p)):
        return f"{cmd}: refusing to send '{ref}' (looks like a secrets file)"
    return None


def _data_refs(name: str, value: str) -> list[str]:
    """Local files a curl data option would upload (-d @file, -F name=@file, ...)."""
    if value.startswith("@"):
        return [value[1:]]
    if name in ("-F", "--form"):
        m = re.match(r"^[^=]*=[@<]([^;]+)", value)
        return [m.group(1)] if m else []
    if name == "--data-urlencode":
        m = re.match(r"^[^=@]*@(.+)$", value)
        return [m.group(1)] if m else []
    return []


def _check_fetch(cmd: str, args: list[str], root: Path) -> str | None:
    w_short, w_long = _FETCH_WRITES[cmd]
    s_short, s_long = _FETCH_SENDS[cmd]
    d_short, d_long = _CURL_DATA if cmd == "curl" else ("", set())
    writes = {"-" + c for c in w_short} | set(w_long)
    sends = {"-" + c for c in s_short} | set(s_long)
    data = {"-" + c for c in d_short} | set(d_long)
    opts, _ = _getopt(args, short_val=w_short + s_short + d_short, long_val=set(w_long) | set(s_long) | set(d_long))
    for name, value in opts:
        if value is None:
            continue
        if name in writes:
            if value in ("-", "/dev/null"):
                continue
            _, err = _resolve_in_root(value, root)
            if err:
                return f"{cmd}: {err}"
        elif name in sends:
            reason = _check_send_path(cmd, value, root)
            if reason:
                return reason
        elif name in data:
            for ref in _data_refs(name, value):
                reason = _check_send_path(cmd, ref, root)
                if reason:
                    return reason
    return None


def _check_tar(args: list[str], root: Path) -> str | None:
    targs = list(args)
    if targs and not targs[0].startswith("-") and re.fullmatch(r"[A-Za-z]+", targs[0]):
        targs[0] = "-" + targs[0]  # old-style `tar xf a.tar`
    opts, pos = _getopt(targs, short_val="CfT", long_val={"--directory", "--file", "--files-from"})
    names = {n for n, _ in opts}
    if names & {"-P", "--absolute-names"}:
        return "tar: absolute names are not allowed"
    if names & {"-I", "-F", "--use-compress-program", "--to-command", "--checkpoint-action", "--rsh-command",
                "--info-script", "--new-volume-script"}:
        return "tar: that option can run arbitrary commands"
    for n, v in opts:
        if n in {"-C", "-f", "-T", "--directory", "--file", "--files-from"} and v and v != "-":
            _, err = _resolve_in_root(v, root)
            if err:
                return f"tar: {err}"
    if names & {"-c", "--create", "-r", "--append", "-u", "--update"}:
        for a in pos:
            reason = _check_read_path("tar", a, root)
            if reason:
                return reason
    return None


def _check_git(args: list[str], root: Path) -> str | None:
    rest = list(args)
    while rest and rest[0].startswith("-"):
        flag = rest[0]
        if flag in ("-C", "-c") and len(rest) > 1:
            value = rest[1]
            if flag == "-C":
                _, err = _resolve_in_root(value, root)
                if err:
                    return f"git -C: {err}"
            else:
                key = value.split("=", 1)[0]
                if _RISKY_GIT_CONFIG.match(key.lower()):
                    return f"git -c {key}: can make git run arbitrary commands"
            rest = rest[2:]
        elif flag.startswith(("--git-dir", "--work-tree")):
            name, eq, value = flag.partition("=")
            if not eq and len(rest) > 1:
                value = rest[1]
            if value:
                _, err = _resolve_in_root(value, root)
                if err:
                    return f"git {name}: {err}"
            rest = rest[1:] if eq else rest[2:]
        else:
            rest = rest[1:]
    sub = rest[0] if rest else ""
    sargs = rest[1:]
    flags = [a for a in sargs if a.startswith("-")]
    shorts = set("".join(f[1:] for f in flags if not f.startswith("--")))
    longs = {f.split("=")[0] for f in flags if f.startswith("--")}
    positional = [a for a in sargs if not a.startswith("-")]
    first = positional[0] if positional else ""

    if sub == "clean":
        return "git clean deletes untracked files; ask the user to run it"
    if sub == "reset" and "--hard" in sargs:
        return "git reset --hard discards uncommitted work; ask the user to run it"
    if sub == "push":
        if "f" in shorts or "d" in shorts or any(l.startswith("--force") for l in longs) \
                or longs & {"--mirror", "--prune", "--delete"}:
            return "force-pushing or deleting remote refs is not allowed"
        if any(a.startswith(("+", ":")) for a in positional):
            return "a +refspec or :ref push can overwrite or delete remote history; ask the user to run it"
    if sub == "checkout" and ("--" in sargs or "." in sargs or "f" in shorts or "--force" in longs):
        return "git checkout can discard uncommitted changes; ask the user to run it"
    if sub == "restore":
        staged_only = ("--staged" in longs or "S" in shorts) and not ("--worktree" in longs or "W" in shorts)
        if not staged_only:
            return "git restore discards uncommitted changes; ask the user to run it"
    if sub == "switch" and ("f" in shorts or longs & {"--force", "--discard-changes"}):
        return "git switch --force discards uncommitted changes; ask the user to run it"
    if sub == "branch" and ("D" in shorts or ("d" in shorts and "f" in shorts)
                            or ("--delete" in longs and "--force" in longs)
                            or ("--force" in longs and "d" in shorts)):
        return "force-deleting a branch is not allowed; ask the user to run it"
    if sub == "stash" and first in {"drop", "clear"}:
        return "git stash drop/clear discards saved work; ask the user to run it"
    if sub == "reflog" and first in {"expire", "delete"}:
        return "git reflog expire/delete destroys recovery points; ask the user to run it"
    if sub == "gc" and any(l.startswith("--prune") for l in longs):
        return "git gc --prune permanently deletes objects; ask the user to run it"
    if sub in {"prune", "filter-branch", "filter-repo", "update-ref"}:
        return f"git {sub} rewrites or destroys history; ask the user to run it"
    return None


def _strip_heredocs(command: str) -> str:
    """Drop heredoc bodies (they're data, e.g. file contents) unless the same line
    feeds them to an interpreter, in which case the body is code and stays checked."""
    out: list[str] = []
    pending: tuple[str, bool, bool] | None = None  # (delimiter, dash form, keep body)
    for line in command.split("\n"):
        if pending:
            delim, dash, keep = pending
            if (line.lstrip("\t") if dash else line) == delim:
                pending = None
            if keep:
                out.append(line)
            continue
        out.append(line)
        m = _HEREDOC.search(line)
        if m:
            pending = (m.group(2), line[m.start():].startswith("<<-"), bool(_RUNS_STDIN.search(line)))
    return "\n".join(out)


def _download_outputs(command: str, root: Path) -> set[Path]:
    """Files a curl/wget in this command line would create (so we can refuse to run them)."""
    out: set[Path] = set()
    try:
        tokens = _tokenize(_strip_heredocs(command))
    except ValueError:
        return out
    prev = ""
    for _, words in _segments(tokens):
        start = _command_start(words)
        cmd = os.path.basename(start[0]) if start else ""
        args = start[1:] if start else []
        names: list[str] = []
        if cmd in NETWORK_GET:
            w_short, w_long = _FETCH_WRITES[cmd]
            writes = {"-" + c for c in w_short} | set(w_long)
            opts, pos = _getopt(args, short_val=w_short, long_val=set(w_long))
            explicit = [v for n, v in opts if n in writes and v]
            prefix = next((v for n, v in opts if n in {"-P", "--directory-prefix", "--output-dir"} and v), "")
            names += explicit
            named_output = any(n in {"-O", "--output-document"} and v for n, v in opts) if cmd == "wget" else False
            remote_name = any(n in {"-O", "--remote-name", "--remote-name-all"} for n, _ in opts) if cmd == "curl" else not named_output
            if remote_name:
                for u in pos:
                    if "://" in u:
                        base = os.path.basename(urlparse(u).path.rstrip("/")) or "index.html"
                        names.append(os.path.join(prefix, base) if prefix else base)
            for i, t in enumerate(words[:-1]):
                if t in REDIRECTS:
                    names.append(words[i + 1])
        elif cmd == "tee" and prev in NETWORK_GET:
            names += [a for a in args if not a.startswith("-")]
        for n in names:
            if n and "$" not in n and n not in ("-", "/dev/null"):
                out.add((root / n).resolve())
        prev = cmd
    return out


def _check_words(sep: str, prev_cmd: str, words: list[str], root: Path, depth: int, tainted: set[Path]) -> str | None:
    # redirections must stay inside the workspace (or /dev/null)
    stdin_files: list[Path | None] = []
    for i, t in enumerate(words[:-1]):
        target = words[i + 1]
        if t in REDIRECTS:
            if target == "/dev/null" or (t == ">&" and (target.isdigit() or target == "-")):
                continue
            p, err = _resolve_in_root(target, root)
            if err:
                return f"redirect target: {err}"
            if _is_session_file(p):
                return f"redirect target: {p.name} is the session log managed by ai"
        elif t == "<" and target != "/dev/null":
            p, err = _resolve_in_root(target, root)
            if err:
                return f"input redirect: {err}"
            stdin_files.append(p)

    if "env" in {os.path.basename(w) for w in words} and any(
            w == "-S" or w.startswith(("-S", "--split-string")) for w in words):
        return "env -S runs a command string that can't be verified"

    words = _command_start(words)
    if not words:
        return None
    head = words[0]
    cmd, args = os.path.basename(head), words[1:]
    flagless = [a for a in args if not a.startswith("-")]

    # `x=rm; $x -rf /`, `rm${IFS}-rf${IFS}/`, `.*` ... the shell expands these after we looked
    if head not in ("[", "[[") and (re.search(r"[$*?{\[]", head) or head.startswith("~")):
        return f"can't verify command '{head}' (variable, glob or expansion in the command name)"

    if cmd in ALWAYS_BLOCKED or cmd.startswith("mkfs"):
        return f"'{cmd}' is not allowed"

    interp = _is_interpreter(cmd)
    if sep.startswith("|") and interp:
        if prev_cmd in NETWORK_GET:
            return "piping a download into a shell or interpreter is not allowed"
        o, p = _getopt(args, short_val="cmeEpWXr")
        if not any(n in ("-c", "-m", "-e", "-E", "-p") for n, _ in o) and not [x for x in p if x != "-"]:
            return "piping into a bare interpreter would run unreviewed code; write it to a file first"
    if cmd in SHELLS:
        for i, a in enumerate(args):
            if a.startswith("-") and not a.startswith("--") and "c" in a and i + 1 < len(args):
                return shell_danger(args[i + 1], root, depth + 1, tainted)
    if cmd == "eval":
        return shell_danger(" ".join(args), root, depth + 1, tainted)
    if cmd == "xargs":
        if any(os.path.basename(a) in (MUTATORS | ALWAYS_BLOCKED | SHELLS) for a in args):
            return "xargs with file-modifying commands is not allowed (paths can't be verified); name the paths or use find -delete"

    # never run something a download command produced (here or earlier in the session)
    if interp or cmd in SOURCERS:
        _, scripts = _getopt(args, short_val="cmeEpWXr")
        for a in scripts:
            p, _ = _resolve_in_root(a, root)
            if p is not None and p in tainted:
                return f"'{a}' was downloaded by a command; ask the user to review and run it"
        if any(f is not None and f in tainted for f in stdin_files):
            return "refusing to feed a downloaded file to an interpreter; ask the user to review and run it"
    if "/" in head:
        p, _ = _resolve_in_root(head, root)
        if p is not None and p in tainted:
            return f"'{head}' was downloaded by a command; ask the user to review and run it"

    if cmd in {"cd", "pushd"}:
        if len(flagless) != 1 or flagless[0] == "-":
            return "cd must name one directory inside the workspace"
        _, err = _resolve_in_root(flagless[0], root)
        if err:
            return err
    if cmd in MUTATORS:
        targets = flagless[1:] if cmd in {"chmod", "chown", "chgrp"} else flagless
        for raw in targets:
            expanded = _brace_expand(raw)
            if expanded is None:
                return f"{cmd}: too many brace alternatives in '{raw}' to verify"
            for a in expanded:
                if cmd in DESTRUCTIVE and _dot_glob(a):
                    return f"{cmd}: '{a}' is a glob that could match .git or other hidden files"
                p, err = _resolve_in_root(a, root)
                if err:
                    return f"{cmd}: {err}"
                if cmd in DESTRUCTIVE and (p == root or p.is_relative_to(root / ".git")):
                    return f"{cmd}: refusing to modify the workspace root or .git"
                if _is_session_file(p):
                    return f"{cmd}: {p.name} is the session log managed by ai"
    if cmd == "dd":
        for a in args:
            if a.startswith(("of=", "if=")):
                _, err = _resolve_in_root(a[3:], root)
                if err:
                    return f"dd: {err}"
    if cmd == "kill" and any(a in {"-1", "0"} for a in args):
        return "kill -1 / kill 0 would signal every process"

    if cmd in READERS:
        opts, pos = _getopt(args, short_val=_READER_SHORT_VALUES.get(cmd, ""),
                            long_val=_READER_LONG_VALUES, short_opt="i" if cmd == "sed" else "")
        if cmd in PATTERN_FIRST and pos and not any(n in _PATTERN_OPTS for n, _ in opts):
            pos = pos[1:]  # the pattern / script, not a path
        for a in pos:
            reason = _check_read_path(cmd, a, root)
            if reason:
                return reason
    if cmd == "sed":  # sed -i edits files in place
        s_opts, s_pos = _getopt(args, short_val="ef", long_val=_READER_LONG_VALUES, short_opt="i")
        if any(n == "-i" or n.startswith("--in-place") for n, _ in s_opts):
            for a in s_pos:
                p, _ = _resolve_in_root(a, root)
                if p is not None and _is_session_file(p):
                    return f"sed: {p.name} is the session log managed by ai"
    if cmd in NETWORK_GET:
        reason = _check_fetch(cmd, args, root)
        if reason:
            return reason
    if cmd == "tar":
        reason = _check_tar(args, root)
        if reason:
            return reason
    if cmd == "find":
        starts = []
        for a in args:
            if a.startswith("-") or a in {"!", "("}:
                break
            starts.append(a)
        for a in starts or ["."]:
            if "-delete" in args:
                _, err = _resolve_in_root(a, root)
                if err:
                    return f"find -delete: {err}"
            else:
                reason = _check_read_path("find", a, root)
                if reason:
                    return reason
        for i, a in enumerate(args):
            if a in {"-exec", "-execdir", "-ok", "-okdir"}:
                r = _check_words("", "", args[i + 1:], root, depth, tainted)
                if r:
                    return r
    if cmd == "git":
        return _check_git(args, root)
    return None


def shell_danger(command: str, root: Path, depth: int = 0, downloaded: set[Path] | None = None) -> str | None:
    """Return a reason if the command looks dangerous, else None.
    `downloaded` holds files that earlier curl/wget commands in this session created."""
    if depth > 3:
        return "commands nested too deeply to verify"
    command = _strip_heredocs(command)
    if re.search(r":\s*\(\s*\)\s*\{", command):
        return "fork bomb"
    try:
        tokens = _tokenize(command)
    except ValueError as e:
        return f"can't parse command ({e})"
    tainted = set(downloaded or ()) | _download_outputs(command, root)
    prev_cmd = ""
    for sep, words in _segments(tokens):
        reason = _check_words(sep, prev_cmd, words, root, depth, tainted)
        if reason:
            return reason
        start = _command_start(words)
        prev_cmd = os.path.basename(start[0]) if start else ""
    return None


# Environment variables commands may see. Everything else (API keys, cloud
# credentials, tokens...) is withheld. Extend with AI_PASSTHROUGH_ENV=NAME1,NAME2.
SAFE_ENV = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TERM", "TMPDIR", "TZ",
            "VIRTUAL_ENV", "COLORTERM", "NO_COLOR"}


def _clean_env() -> dict[str, str]:
    keep = SAFE_ENV | {n.strip() for n in os.getenv("AI_PASSTHROUGH_ENV", "").split(",") if n.strip()}
    return {k: v for k, v in os.environ.items() if k in keep or k.startswith("LC_")}


def _as_int(value, name: str, minimum: int = 0) -> int | None:
    """Models sometimes send numbers as strings ("10") or floats (10.0); accept both."""
    if value is None or value == "":
        return None
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be an integer, got {value!r}")
    if n < minimum:
        raise ValueError(f"'{name}' must be >= {minimum}, got {n}")
    return n


class Toolbox:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.downloaded: set[Path] = set()  # files created by curl/wget in this session; never executed
        self.protected: set[Path] = set()  # files the tools must never modify (the session log)
        self._proc: subprocess.Popen | None = None  # the run_shell command currently running, if any
        self._killed = False
        self.tools: dict[str, Tool] = {t.name: t for t in self._build()}

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def _path(self, p: str) -> Path:
        full = (self.root / p).resolve()
        if not full.is_relative_to(self.root):
            raise ValueError(f"Path escapes workspace: {p}")
        return full

    def execute(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"Error: unknown tool '{name}'"
        try:
            return _clip(str(tool.fn(**args)))
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:  # tool errors go back to the model
            return f"Error: {e}"

    def list_dir(self, path: str = ".") -> str:
        d = self._path(path)
        rows = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        return "\n".join(f"{p.name}{'/' if p.is_dir() else ''}" for p in rows if p.name not in SKIP_DIRS) or "(empty)"

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        """Read a text file. `offset` is the zero-based index of the first line to
        return; `limit` is the max number of lines. With neither, returns the whole
        file. If the result is too big it is cut at a line boundary and a note says
        which offset to use to continue."""
        start = _as_int(offset, "offset") or 0
        count = _as_int(limit, "limit", minimum=1)
        lines = self._path(path).read_text(errors="replace").splitlines(keepends=True)
        total = len(lines)
        if start == 0 and count is None and sum(len(l) for l in lines) <= MAX_OUT:
            return "".join(lines)  # fast path: whole small file, unchanged behaviour
        if start >= total:
            return f"(offset {start} is past the end of the file; it has {total} lines)"
        end = total if count is None else min(total, start + count)

        out: list[str] = []
        size = 0
        i = start
        while i < end:
            # always emit at least one line so a huge single line can't stall paging
            if out and size + len(lines[i]) > MAX_OUT - 200:
                break
            out.append(lines[i])
            size += len(lines[i])
            i += 1
        text = "".join(out)
        if text and not text.endswith("\n"):
            text += "\n"
        if i < end or end < total:
            text += f"... [showing lines {start}-{i - 1} of {total}; call read_file with offset={i} to continue]"
        elif start > 0:
            text += f"... [showing lines {start}-{i - 1} of {total}; end of file]"
        return text

    def protect(self, path) -> None:
        self.protected.add(Path(path).resolve())

    def _check_writable(self, p: Path) -> None:
        if p.name.startswith(SESSION_PREFIX) or p in self.protected:
            raise ValueError(f"{p.name} is the session log managed by ai; tools can't modify it")

    def write_file(self, path: str, content: str) -> str:
        p = self._path(path)
        self._check_writable(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = self._path(path)
        self._check_writable(p)
        text = p.read_text()
        n = text.count(old)
        if n == 0:
            return "Error: 'old' text not found"
        if n > 1:
            return f"Error: 'old' text matches {n} places; include more context"
        p.write_text(text.replace(old, new))
        return f"Edited {path}"

    def search(self, pattern: str, path: str = ".") -> str:
        base, hits = self._path(path), []
        files = [base] if base.is_file() else base.rglob("*")
        for f in files:
            if not f.is_file() or any(part in SKIP_DIRS for part in f.parts):
                continue
            if f.name.startswith(SESSION_PREFIX) and f != base:  # our own logs just repeat everything else
                continue
            try:
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if pattern in line:
                        hits.append(f"{f.relative_to(self.root)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= 100:
                            return "\n".join(hits) + "\n... (100 hit limit)"
            except (UnicodeDecodeError, OSError):
                continue
        return "\n".join(hits) or "No matches"

    def _validate_shell(self, command: str) -> str | None:
        return shell_danger(command, self.root, downloaded=self.downloaded)

    def run_shell(self, command: str) -> str:
        reason = self._validate_shell(command)
        if reason:
            return f"Blocked: {reason}"
        self.downloaded |= _download_outputs(command, self.root)
        proc = subprocess.Popen(
            command, shell=True, cwd=self.root, env=_clean_env(), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        self._proc, self._killed = proc, False
        try:
            out, err = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # kill the whole process group, not just the shell
            proc.communicate()
            return "Error: command timed out after 120s and was killed"
        finally:
            self._proc = None
        if self._killed:
            self._killed = False
            return "Stopped by the user: the command was killed."
        return f"exit code {proc.returncode}\n{out}{err}"

    def kill_running(self) -> None:
        """Kill the run_shell command that is running right now (called when the user presses stop)."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self._killed = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _build(self) -> list[Tool]:
        s = {"type": "string"}
        n = {"type": "integer"}
        return [
            Tool("list_dir", "List files in a directory.", _params({"path": s}, []), self.list_dir),
            Tool("read_file",
                 "Read a text file. Optionally pass 'offset' (zero-based line number to start at) and "
                 "'limit' (max lines) to read a slice; long files are cut off with a note giving the "
                 "offset to continue from.",
                 _params({"path": s, "offset": n, "limit": n}, ["path"]), self.read_file),
            Tool("search", "Find lines containing a literal string across files.",
                 _params({"pattern": s, "path": s}, ["pattern"]), self.search),
            Tool("write_file", "Create or overwrite a file.",
                 _params({"path": s, "content": s}, ["path", "content"]), self.write_file, dangerous=True),
            Tool("edit_file", "Replace one exact occurrence of 'old' with 'new' in a file.",
                 _params({"path": s, "old": s, "new": s}, ["path", "old", "new"]), self.edit_file, dangerous=True),
            Tool("run_shell", "Run a shell command in the workspace root.",
                 _params({"command": s}, ["command"]), self.run_shell, dangerous=True,
                 validate=self._validate_shell),
        ]


# --------------------------------------------------------------------------
# Session log: an append-only record of the conversation so it can be resumed
# --------------------------------------------------------------------------
SESSION_FILE = ".ai-session.jsonl"   # lives in the project root
SESSION_PREFIX = ".ai-session"       # tools refuse to modify any file with this prefix (logs and archives)
SESSION_VERSION = 1
LOGGED_KINDS = {"assistant", "tool", "result", "error", "notice"}  # transcript events worth keeping
LOG_REASONING_CHARS = 8000           # the model's reasoning is logged, but only its last part
REPLAY_MAX = 200                     # on resume, show at most this many past entries on screen


class SessionLocked(Exception):
    """Another ai process is using this session log."""


class SessionVersionError(Exception):
    """The log was written by a newer ai than this one."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionLog:
    """Append-only JSON-lines record of one session. One JSON object per line:

      {"v":1,"type":"session","id":...,"created":...,"root":...}      header, first line
      {"type":"message","ts":...,"message":{...}}                      one chat message (what the model sees)
      {"type":"update_last","ts":...,"content":...}                    the last user message was extended
      {"type":"transcript","ts":...,"kind":"user|assistant|tool|result|error|notice|reasoning","text":...}
      {"type":"event","ts":...,"name":...,...}                         e.g. supervision toggled, resumed

    Every record is flushed and fsynced as it is written, so killing ai at any moment loses at most a
    half-written last line, which loading ignores. The file is created on the first record (opening ai
    and quitting leaves nothing behind) and is flock'ed while open, so two ai processes can't interleave
    writes. Failures to write never raise: they set `error` and the agent carries on."""

    def __init__(self, path, root=None):
        self.path = Path(path)
        self.root = Path(root) if root else None
        self.id = uuid.uuid4().hex[:12]
        self.error: str | None = None
        self._fh = None
        self._lock = threading.Lock()

    # ---- file handling ---------------------------------------------------------
    def exists(self) -> bool:
        try:
            return self.path.stat().st_size > 0
        except OSError:
            return False

    def _write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._fh.write(line.encode("utf-8"))
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def _open(self) -> None:  # call with self._lock held
        if self._fh is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if fcntl:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise SessionLocked(f"{self.path} is in use by another ai process")
        self._fh = fh
        if fh.seek(0, os.SEEK_END) == 0:
            self._write({"v": SESSION_VERSION, "type": "session", "id": self.id, "created": _now(),
                         "root": str(self.root) if self.root else None})
        else:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":  # a previous run died mid-line: don't glue the next record onto it
                fh.write(b"\n")
                fh.flush()

    def _close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()  # also releases the lock
            finally:
                self._fh = None

    def open(self) -> None:
        """Take the lock on an existing log (raises SessionLocked if another ai has it)."""
        with self._lock:
            self._open()

    def close(self) -> None:
        with self._lock:
            self._close()

    def append(self, type_: str, **fields) -> None:
        with self._lock:
            try:
                self._open()
                self._write({"type": type_, "ts": _now(), **fields})
            except SessionLocked as e:
                self.error = str(e)
            except OSError as e:
                self.error = f"can't write {self.path}: {e}"

    # ---- records -----------------------------------------------------------------
    def message(self, message: dict) -> None:
        self.append("message", message=message)

    def update_last(self, content: str) -> None:
        self.append("update_last", content=content)

    def transcript(self, kind: str, text: str) -> None:
        self.append("transcript", kind=kind, text=text)

    def event(self, name: str, **data) -> None:
        self.append("event", name=name, **data)

    # ---- reading / starting over ---------------------------------------------------
    def load(self) -> tuple[list[dict], int]:
        """All readable records, and how many lines were unreadable (e.g. a torn last line)."""
        records: list[dict] = []
        skipped = 0
        with open(self.path, "rb") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line.decode("utf-8"))
                except ValueError:
                    skipped += 1
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
                else:
                    skipped += 1
        header = next((r for r in records if r.get("type") == "session"), None)
        if header:
            if header.get("v", 1) > SESSION_VERSION:
                raise SessionVersionError(
                    f"{self.path} was written by a newer ai (log version {header['v']}, this one reads "
                    f"{SESSION_VERSION}); use --new to start fresh or --session FILE for another log")
            self.id = header.get("id", self.id)
        return records, skipped

    def archive(self) -> Path | None:
        """Move the current log aside (never delete it) so the next record starts a fresh session."""
        with self._lock:
            if not self.exists():
                self._close()
                return None
            self._open()  # take the lock first: never move a log that another ai is writing
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self.path.with_name(f"{self.path.stem}.{stamp}{self.path.suffix}")
            n = 1
            while dest.exists():
                dest = self.path.with_name(f"{self.path.stem}.{stamp}-{n}{self.path.suffix}")
                n += 1
            os.replace(self.path, dest)
            self._close()
            self.id = uuid.uuid4().hex[:12]
            return dest


@dataclass
class ResumeInfo:
    transcript: list[dict]   # past visible entries, for redrawing the screen
    notes: list[str]         # things the user should know (e.g. the last run was interrupted)
    created: str
    messages: int
    path: Path


def restore_history(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Rebuild (chat messages, transcript entries) from log records. The system message is not logged:
    it is regenerated, so a newer ai gets its newer instructions."""
    messages: list[dict] = []
    transcript: list[dict] = []
    for rec in records:
        kind = rec.get("type")
        if kind == "message" and isinstance(rec.get("message"), dict):
            messages.append(rec["message"])
        elif kind == "update_last" and messages and messages[-1].get("role") == "user":
            messages[-1]["content"] = str(rec.get("content", ""))
        elif kind == "transcript":
            transcript.append(rec)
    return messages, transcript


def repair_interrupted_tool_calls(messages: list[dict]) -> list[dict]:
    """If ai was closed while tool calls were running, the log ends with an assistant message whose calls
    have no (or only some) results. A model API rejects that history, so return stand-in results for the
    missing ones (the caller appends them)."""
    i = len(messages) - 1
    while i >= 0 and messages[i].get("role") == "tool":
        i -= 1
    if i < 0 or messages[i].get("role") != "assistant" or not messages[i].get("tool_calls"):
        return []
    answered = {m.get("tool_call_id") for m in messages[i + 1:]}
    return [{"role": "tool", "tool_call_id": c.get("id", ""),
             "content": "Interrupted: ai was closed before this tool call finished. It may or may not have "
                        "completed; check before repeating it."}
            for c in messages[i]["tool_calls"] if c.get("id", "") not in answered]


# --------------------------------------------------------------------------
# Agent loop: model -> tool calls -> results -> model, until it answers
# --------------------------------------------------------------------------
SYSTEM = """You are ai, a careful software engineering agent working in {root}.
Use tools to inspect the project before changing it. Make small, focused edits
with edit_file when possible. After changes, verify them (run tests or the code).
Be concise. When the task is done, say what you changed."""

# Emit kinds: assistant / tool / result / error go to the transcript;
# "status" (what the agent is doing right now), "progress" (live detail while the
# model is generating, e.g. "bots/page.tsx \u00b7 14,203 chars") and "hint" (a short
# note about the current step) only feed the live activity panel in the TUI.
Emit = Callable[[str, str], None]       # (kind, text)
Approve = Callable[[str, dict], bool]   # (tool name, args) -> allowed?

TOOL_LABELS = {
    "list_dir": "Listing a directory",
    "read_file": "Reading a file",
    "search": "Searching the project",
    "write_file": "Writing a file",
    "edit_file": "Editing a file",
    "run_shell": "Running a command",
}


def _one_line(text, limit: int = 100) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def tool_label(name: str) -> str:
    return TOOL_LABELS.get(name, f"Using {name}")


def describe_call(name: str, args) -> str:
    """One-line, human-readable description of a tool call for the activity panel."""
    a = args if isinstance(args, dict) else {}
    if name == "list_dir":
        text = f"Listing {a.get('path') or '.'}"
    elif name == "read_file":
        text = f"Reading {a.get('path') or '?'}"
        if a.get("offset") not in (None, "", 0):
            text += f" (from offset {a['offset']})"
    elif name == "search":
        text = f"Searching for '{a.get('pattern', '')}'"
        if a.get("path"):
            text += f" in {a['path']}"
    elif name == "write_file":
        text = f"Writing {a.get('path') or '?'}"
    elif name == "edit_file":
        text = f"Editing {a.get('path') or '?'}"
    elif name == "run_shell":
        text = f"Running: {a.get('command', '')}"
    else:
        text = f"Calling {name}"
    return _one_line(text)


def summarize_reasoning(text) -> str:
    """First meaningful line of a reasoning model's thinking, as a short hint."""
    if not isinstance(text, str):
        return ""
    for line in text.splitlines():
        line = line.strip().strip("*#-> `").strip()
        if line:
            return _one_line(line, 110)
    return ""


def format_elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60:02d}s"


class Agent:
    def __init__(self, llm: LMStudio, toolbox: Toolbox, max_steps: int = 25, max_continues: int = 5,
                 session: SessionLog | None = None):
        self.llm, self.toolbox, self.max_steps = llm, toolbox, max_steps
        self.session = session  # if set, everything is recorded so the conversation can be resumed
        self.max_continues = max_continues  # consecutive generation timeouts we retry (0 = no limit)
        self._stopped = threading.Event()
        self.messages: list[dict] = []
        self.reset()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM.format(root=self.toolbox.root)}]

    # ---- history + session log ---------------------------------------------------------
    def _append(self, message: dict) -> None:
        self.messages.append(message)
        if self.session:
            self.session.message(message)

    def _update_last_user(self, content: str) -> None:
        self.messages[-1]["content"] = content
        if self.session:
            self.session.update_last(content)

    def _logging(self, emit: Emit) -> Emit:
        if not self.session:
            return emit

        def logged(kind: str, text: str) -> None:
            if kind in LOGGED_KINDS:
                self.session.transcript(kind, text)
            emit(kind, text)

        return logged

    def start_new_session(self) -> Path | None:
        """Forget the conversation. The old log is moved aside (not deleted); returns where it went."""
        self.reset()
        return self.session.archive() if self.session else None

    def resume(self) -> ResumeInfo | None:
        """Restore the conversation from the session log, if there is one. Raises SessionLocked if another
        ai process is using it. If ai was closed mid-run, the history is repaired so it is valid again."""
        if self.session is None or not self.session.exists():
            return None
        self.session.open()
        records, skipped = self.session.load()
        messages, transcript = restore_history(records)
        self.reset()
        self.messages.extend(messages)
        repairs = repair_interrupted_tool_calls(self.messages)
        for r in repairs:
            self._append(r)
        last = self.messages[-1]["role"]
        notes: list[str] = []
        if repairs:
            notes.append(f"{len(repairs)} tool call(s) were still running when ai was closed; the model "
                         "has been told they may not have finished.")
        elif last == "user":
            notes.append("The previous run was interrupted before the model replied.")
        elif last == "tool":
            notes.append("The previous run was interrupted before the model could react to the last tool result.")
        if last in ("user", "tool"):
            notes.append('Type "continue" to pick up where it left off.')
        if skipped:
            notes.append(f"{skipped} unreadable line(s) in the session log were skipped.")
        self.session.event("resumed", messages=len(messages), repaired=len(repairs))
        header = next((r for r in records if r.get("type") == "session"), {})
        return ResumeInfo(transcript, notes, header.get("created", "?"), len(messages), self.session.path)

    def stopped(self) -> bool:
        return self._stopped.is_set()

    def stop(self) -> None:
        """End the current run now (called from the UI thread). Aborts the reply being generated
        (closing the stream is what makes LM Studio stop) and kills a running shell command."""
        self._stopped.set()
        abort = getattr(self.llm, "abort", None)
        if abort:
            abort("stopped")
        self.toolbox.kill_running()

    def _add_continue(self) -> None:
        """Our retry mechanism: send "continue". If the last message is the user's, fold it into that
        message so user/assistant roles still alternate (some chat templates insist on it)."""
        last = self.messages[-1] if self.messages else None
        if last and last.get("role") == "user":
            self._update_last_user(f"{last['content']}\n\ncontinue")
        else:
            self._append({"role": "user", "content": "continue"})

    def run(self, user_text: str, emit: Emit, approve: Approve,
            unlimited: Callable[[], bool] | None = None) -> None:
        """Run until the model answers without calling tools. The max_steps cap is
        skipped while `unlimited()` is true; it's re-checked before every step, so
        flipping it mid-run takes effect immediately.

        A reply that runs past the generation limit is stopped, thrown away, and "continue" is sent
        as a retry (up to max_continues times in a row). stop() ends the run."""
        emit = self._logging(emit)
        self._stopped.clear()
        if self.session:
            self.session.transcript("user", user_text)
        last = self.messages[-1] if self.messages else None
        if last and last.get("role") == "user":  # the previous run died before the model replied: keep roles alternating
            self._update_last_user(f"{last['content']}\n\n{user_text}")
        else:
            self._append({"role": "user", "content": user_text})
        steps = continues = 0
        while True:
            if self._stopped.is_set():
                emit("notice", "Stopped.")
                return
            if steps >= self.max_steps and not (unlimited and unlimited()):
                emit("error", f"Stopped after {steps} steps.")
                return
            steps += 1
            emit("status", "Thinking")
            shown = {"label": "Thinking"}
            streamed = {"any": False}

            def progress(label: str, detail: str) -> None:
                if label != shown["label"]:
                    shown["label"] = label
                    emit("status", label)
                emit("progress", detail)

            def on_text(channel: str, text: str) -> None:
                streamed["any"] = True
                emit("stream_" + channel, text)

            try:
                msg = self.llm.chat(self.messages, self.toolbox.schemas(), progress=progress,
                                    on_text=on_text, cancelled=self._stopped.is_set)
            except GenerationStopped:
                if streamed["any"]:
                    emit("stream_end", "aborted")
                emit("notice", "Stopped.")
                return
            except GenerationTimeout as e:
                if streamed["any"]:
                    emit("stream_end", "aborted")
                continues += 1
                if self.max_continues and continues > self.max_continues:
                    emit("error", f"{e} That happened {continues} times in a row, so I gave up; "
                                  "send a message to try again.")
                    return
                limit = self.max_continues or "no"
                emit("notice", f"{e} Sent \"continue\" to retry (retry {continues}, {limit} limit).")
                self._add_continue()
                steps -= 1  # a retry isn't progress, so it doesn't eat into the step limit
                continue
            continues = 0
            if streamed["any"]:
                emit("stream_end", "")
            if msg.get("finish_reason") == "length":
                emit("error", "The reply hit the model's token limit and was cut off; "
                              "raise the context/max tokens in LM Studio.")
            reasoning_text = msg.get("reasoning_content") or msg.get("reasoning")
            if self.session and isinstance(reasoning_text, str) and reasoning_text.strip():
                self.session.transcript("reasoning", reasoning_text[-LOG_REASONING_CHARS:])
            hint = summarize_reasoning(reasoning_text)
            if hint:
                emit("hint", hint)
            calls = msg.get("tool_calls") or []
            entry: dict = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                entry["tool_calls"] = calls
            self._append(entry)
            if entry["content"].strip():
                emit("assistant", entry["content"])
            if not calls:
                return
            for call in calls:
                if self._stopped.is_set():  # keep the history valid: every tool call needs a result
                    self._append({"role": "tool", "tool_call_id": call.get("id", ""),
                                  "content": "Cancelled: the user stopped the run."})
                    continue
                self._run_call(call, emit, approve)

    def _run_call(self, call: dict, emit: Emit, approve: Approve) -> None:
        fn = call["function"]
        name = fn["name"]
        try:
            raw = fn.get("arguments") or "{}"
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            result = "Error: arguments were not valid JSON"
        else:
            emit("tool", f"{name} {json.dumps(args)[:200]}")
            emit("hint", describe_call(name, args))
            tool = self.toolbox.tools.get(name)
            reason = None
            if tool and tool.validate:
                try:
                    reason = tool.validate(**args)
                except TypeError:
                    pass  # bad arguments get reported by execute()
            if reason:
                result = f"Blocked: {reason}. Don't retry this; pick a safer approach or ask the user."
                emit("error", f"blocked {name}: {reason}")
            else:
                allowed = True
                if tool and tool.dangerous:
                    allowed = approve(name, args)
                if not allowed:
                    result = "User denied this action."
                    emit("error", f"denied: {name}")
                else:
                    emit("status", tool_label(name))
                    result = self.toolbox.execute(name, args)
                    emit("result", result[:400])
        self._append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})


# --------------------------------------------------------------------------
# Textual TUI
# --------------------------------------------------------------------------
# Tools that put text on the system clipboard, tried in order. Textual's own copy uses the
# OSC 52 terminal escape, which some terminals (e.g. macOS Terminal.app) ignore.
CLIPBOARD_TOOLS = (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"],
                   ["xsel", "--clipboard", "--input"], ["clip.exe"])


def _copy_with_system_tool(text: str) -> bool:
    for cmd in CLIPBOARD_TOOLS:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode(), timeout=2, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (OSError, subprocess.SubprocessError):
                continue
    return False


_KIND_STYLES = {"tool": "yellow", "result": "dim", "error": "bold red", "notice": "bold yellow"}
LIVE_MAX_CHARS = 6000  # a live (still streaming) block shows at most this much of the newest text
_LIVE_STYLES = {"reasoning": "dim italic", "answer": "", "tool": "dim yellow"}


def _tail(text: str, cap: int = LIVE_MAX_CHARS) -> str:
    if len(text) <= cap:
        return text
    return f"{ELLIPSIS} ({len(text) - cap:,} earlier chars not shown)\n" + text[-cap:]


class Transcript(VerticalScroll):
    """The scrolling conversation log.

    One widget per entry (a Static for plain text, Textual's Markdown for assistant replies) instead
    of a RichLog, because these support mouse text selection (drag to select, double-click for one
    entry, triple-click for everything, ctrl+c to copy) and re-wrap by themselves when the terminal is
    resized. It never takes keyboard focus, so selecting in it doesn't pull typing away from the prompt."""

    can_focus = False

    def __init__(self, **kw):
        super().__init__(**kw)
        self.entries: list[str] = []  # plain text of everything added so far
        self._live: dict[str, Static] = {}  # blocks being streamed right now, by channel
        self._live_text: dict[str, str] = {}

    def on_mount(self) -> None:
        self.anchor()  # follow new output, until the user scrolls up

    def add(self, content, markdown: bool = False) -> None:
        if markdown:
            self.entries.append(str(content))
            self.mount(MarkdownView(str(content)))
        else:
            text = content if isinstance(content, Text) else Text(str(content))
            self.entries.append(text.plain)
            self.mount(Static(text))

    def stream(self, channel: str, text: str) -> None:
        """Show text that is still being generated ("reasoning", "answer" or "tool"); call again with the
        longer text as it grows. Only the newest LIVE_MAX_CHARS are shown so a runaway reply stays cheap."""
        shown = _tail(text)
        self._live_text[channel] = shown
        widget = self._live.get(channel)
        if widget is None:
            widget = self._live[channel] = Static(Text(shown, style=_LIVE_STYLES.get(channel, "")), classes="live")
            self.mount(widget)
        else:
            widget.update(Text(shown, style=_LIVE_STYLES.get(channel, "")))

    def end_stream(self, aborted: bool = False) -> None:
        """The reply is complete. The model's reasoning stays on screen; the live answer and tool-call
        text are removed (the final reply and tool lines replace them). If it was aborted, keep it all
        so you can see what the model was doing."""
        live, texts = self._live, self._live_text
        self._live, self._live_text = {}, {}
        for channel, widget in live.items():
            if aborted or channel == "reasoning":
                self.entries.append(texts.get(channel, ""))
            else:
                widget.remove()

    def reset(self) -> None:
        self.entries.clear()
        self._live, self._live_text = {}, {}
        self.remove_children()


class ApprovalScreen(ModalScreen[bool]):
    BINDINGS = [Binding("y", "allow", "Allow"), Binding("n,escape", "deny", "Deny")]
    DEFAULT_CSS = """
    ApprovalScreen { align: center middle; }
    #box { width: 80%; max-height: 80%; border: thick $warning; background: $surface; padding: 1 2; }
    #detail { width: 100%; height: auto; max-height: 20; overflow-x: hidden; overflow-y: auto; margin-bottom: 1; }
    #buttons { height: 3; align-horizontal: center; }
    """

    def __init__(self, name: str, args: dict):
        super().__init__()
        self.tool_name, self.args = name, args

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Static(Text(f"Allow {self.tool_name}?", style="bold yellow"))
            yield Static(Text(json.dumps(self.args, indent=2)), id="detail")
            with Horizontal(id="buttons"):
                yield Button("Allow (y)", variant="success", id="allow")
                yield Button("Deny (n)", variant="error", id="deny")

    def on_button_pressed(self, e: Button.Pressed) -> None:
        self.dismiss(e.button.id == "allow")

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class ActivityPanel(Static):
    """Live 'what is the agent doing' panel: spinner, current activity, elapsed
    time, and a short list of hints about recent steps. Click it (or press
    ctrl+t) to collapse/expand. Only visible while the agent is working."""

    SPINNER = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
    MAX_HINTS = 6
    DEFAULT_CSS = """
    ActivityPanel { display: none; height: auto; max-height: 12; padding: 0 1; border-top: solid $primary-darken-2; }
    """

    def __init__(self, **kw):
        super().__init__("", **kw)
        self.label = "Thinking"
        self.detail = ""  # live progress while the model generates, e.g. "page.tsx \u00b7 14,203 chars"
        self.hints: list[str] = []
        self.expanded = True
        self.supervised = True
        self._t0 = 0.0
        self._spin = 0
        self._ticker = None

    def start(self) -> None:
        self.label, self.detail, self.hints = "Thinking", "", []
        self._t0, self._spin = time.monotonic(), 0
        if self._ticker is None:
            self._ticker = self.set_interval(0.1, self._tick)
        else:
            self._ticker.resume()
        self.display = True
        self._redraw()

    def stop(self) -> None:
        if self._ticker is not None:
            self._ticker.pause()
        self.display = False

    def set_label(self, label: str) -> None:
        self.label, self.detail = label, ""
        self._redraw()

    def set_detail(self, detail: str) -> None:
        self.detail = detail
        self._redraw()

    def add_hint(self, hint: str) -> None:
        if hint and (not self.hints or self.hints[-1] != hint):
            self.hints.append(hint)
        self._redraw()

    def toggle(self) -> None:
        self.expanded = not self.expanded
        self._redraw()

    def set_supervised(self, supervised: bool) -> None:
        self.supervised = supervised
        self._redraw()

    def on_click(self) -> None:
        self.toggle()

    def _tick(self) -> None:
        self._spin += 1
        self._redraw()

    def _redraw(self) -> None:
        self.update(self.build_text())

    def build_text(self) -> Text:
        t = Text()
        t.append(self.SPINNER[self._spin % len(self.SPINNER)] + " ", style="bold cyan")
        t.append(self.label, style="bold")
        if self.detail:
            t.append("  " + self.detail, style="cyan")
        t.append(f"  {format_elapsed(time.monotonic() - self._t0)}  ", style="dim")
        t.append("\u25be" if self.expanded else "\u25b8", style="dim")
        if not self.supervised:
            t.append("  UNSUPERVISED", style="bold red")
        if self.expanded and self.hints:
            shown = self.hints[-self.MAX_HINTS:]
            older = len(self.hints) - len(shown)
            if older:
                t.append(f"\n  \u2026 {older} earlier step{'s' if older != 1 else ''}", style="dim")
            for i, h in enumerate(shown):
                t.append("\n  \u2022 " + h, style="dim" if i < len(shown) - 1 else "")
        return t


class AiApp(App):
    TITLE = "ai"
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+k", "stop", "Stop", priority=True),
        Binding("ctrl+t", "toggle_activity", "Activity"),
        Binding("ctrl+s", "toggle_supervision", "Supervision: on"),
        Binding("pageup", "log_page_up", "Scroll up", show=False),
        Binding("pagedown", "log_page_down", "Scroll down", show=False),
    ]
    CSS = """
    #log { height: 1fr; padding: 0 1; overflow-x: hidden; }
    #log > Markdown { margin: 0; padding: 0; }
    #prompt { dock: bottom; }
    """

    def __init__(self, agent: Agent, initial_prompt: str | None = None, resume: ResumeInfo | None = None):
        super().__init__()
        self.agent = agent
        self.initial_prompt = initial_prompt
        self.resume = resume  # what to redraw from the session log, if this is a resumed session
        self._session_error_shown: str | None = None
        self.busy = False
        # True (default): write_file / edit_file / run_shell ask y/n and the agent stops
        # after max_steps. False: they run without asking and the step limit is lifted.
        self.supervision = True

    def compose(self) -> ComposeResult:
        yield Header()
        yield Transcript(id="log")
        yield ActivityPanel(id="activity")
        yield Input(placeholder="Ask ai to build something...  (/clear to reset)", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = str(self.agent.toolbox.root)
        self.query_one(Input).focus()
        self._write(Text(f"ai ready. Workspace: {self.agent.toolbox.root}\n"
                         "Drag to select text; ctrl+c copies the selection.", style="dim"))
        if self.resume:
            self._replay(self.resume)
        if self.initial_prompt:
            self.submit(self.initial_prompt)

    def _write(self, renderable) -> None:
        self.query_one(Transcript).add(renderable)

    def _write_markdown(self, text: str) -> None:
        self.query_one(Transcript).add(text, markdown=True)

    def action_log_page_up(self) -> None:  # the prompt has focus, so page keys are routed to the transcript
        self.query_one(Transcript).scroll_page_up(animate=False)

    def action_log_page_down(self) -> None:
        self.query_one(Transcript).scroll_page_down(animate=False)

    def copy_to_clipboard(self, text: str) -> None:
        super().copy_to_clipboard(text)  # OSC 52: iTerm2, kitty, WezTerm, Ghostty, tmux (if enabled), ...
        _copy_with_system_tool(text)     # pbcopy / wl-copy / xclip / xsel for terminals without OSC 52

    def action_clear(self) -> None:
        archived = self.agent.start_new_session()
        self.query_one(Transcript).reset()
        note = f" The previous session was saved as {archived.name}." if archived else ""
        self._write(Text(f"Conversation cleared.{note}", style="dim"))

    def _replay(self, resume: ResumeInfo) -> None:
        """Redraw the earlier part of a resumed session so you can see where you left off."""
        entries = resume.transcript
        shown = entries[-REPLAY_MAX:]
        self._write(Text(f"Resumed session from {resume.created} ({resume.messages} messages). "
                         f"Full log: {resume.path.name}", style="bold green"))
        if len(entries) > len(shown):
            self._write(Text(f"\u2026 {len(entries) - len(shown)} earlier entries not shown", style="dim"))
        for rec in shown:
            kind, text = rec.get("kind"), str(rec.get("text", ""))
            if kind == "assistant":
                self._write_markdown(text)
            elif kind == "user":
                self._write(Text(f"\n> {text}", style="bold cyan"))
            elif kind == "reasoning":
                self._write(Text(_tail(text), style="dim italic"))
            else:
                self._write(Text(("  " if kind == "result" else "") + text, style=_KIND_STYLES.get(kind, "")))
        for note in resume.notes:
            self._write(Text(note, style="bold yellow"))

    def _log_event(self, name: str, **data) -> None:
        if self.agent.session:
            self.agent.session.event(name, **data)

    def check_action(self, action: str, parameters) -> bool | None:
        if action == "stop":  # only live while the agent is working; otherwise ctrl+k is the prompt's own key
            return True if self.busy else None
        return True

    def action_stop(self) -> None:
        if not self.busy:
            return
        self._set_status("Stopping\u2026")
        self._log_event("stop_pressed")
        if isinstance(self.screen, ApprovalScreen):
            self.screen.dismiss(False)
        self.agent.stop()

    def action_toggle_activity(self) -> None:
        self.query_one(ActivityPanel).toggle()

    def action_toggle_supervision(self) -> None:
        self.supervision = not self.supervision
        self._log_event("supervision", on=self.supervision)
        self.title = "ai" if self.supervision else "ai \u2014 UNSUPERVISED"
        self.query_one(ActivityPanel).set_supervised(self.supervision)
        self._set_binding_label("ctrl+s", "Supervision: on" if self.supervision else "Supervision: OFF")
        if self.supervision:
            self._write(Text(f"Supervision ON: file writes, edits and shell commands ask for approval again, "
                             f"and runs stop after {self.agent.max_steps} steps.", style="bold green"))
        else:
            self._write(Text("Supervision OFF: write_file, edit_file and run_shell now run WITHOUT asking, and "
                             "there is no step limit. Blocked shell commands are still blocked. "
                             "Press ctrl+s to turn it back on.", style="bold red"))

    def _set_binding_label(self, key: str, description: str) -> None:
        """Footer labels are static in Textual; swap the Binding to change one at runtime."""
        try:
            # assign a new list: the lists inside key_to_bindings are shared between App instances
            self._bindings.key_to_bindings[key] = [
                dataclass_replace(b, description=description) for b in self._bindings.key_to_bindings[key]]
            self.refresh_bindings()
        except (AttributeError, KeyError):
            pass  # cosmetic only: the title and activity panel still show the state

    def on_input_submitted(self, e: Input.Submitted) -> None:
        text = e.value.strip()
        if not text or self.busy:
            return
        e.input.value = ""
        self.submit(text)

    def submit(self, text: str) -> None:
        if text == "/clear":
            return self.action_clear()
        self._write(Text(f"\n> {text}", style="bold cyan"))
        self.busy = True
        self.sub_title = "thinking..."
        self.query_one(ActivityPanel).start()
        self.refresh_bindings()
        self.run_agent(text)

    # runs in a worker thread; UI updates go through call_from_thread
    @work(thread=True, exclusive=True)
    def run_agent(self, text: str) -> None:
        try:
            self.agent.run(text, self._emit, self._approve, unlimited=lambda: not self.supervision)
        except LLMError as err:
            self._emit("error", str(err))
        except Exception as err:
            self._emit("error", f"{type(err).__name__}: {err}")
        finally:
            self.call_from_thread(self._done)

    def _done(self) -> None:
        self.busy = False
        self.query_one(Transcript).end_stream(aborted=True)  # no-op unless a reply was cut short by an error
        problem = self.agent.session.error if self.agent.session else None
        if problem and problem != self._session_error_shown:
            self._session_error_shown = problem
            self._write(Text(f"Session log problem: {problem}. This session may not be resumable.", style="bold red"))
        self.query_one(ActivityPanel).stop()
        self.refresh_bindings()
        self.sub_title = str(self.agent.toolbox.root)

    def _set_status(self, label: str) -> None:
        self.query_one(ActivityPanel).set_label(label)

    def _add_hint(self, hint: str) -> None:
        self.query_one(ActivityPanel).add_hint(hint)

    def _set_detail(self, detail: str) -> None:
        self.query_one(ActivityPanel).set_detail(detail)

    def _stream(self, channel: str, text: str) -> None:
        self.query_one(Transcript).stream(channel, text)

    def _end_stream(self, aborted: bool) -> None:
        self.query_one(Transcript).end_stream(aborted)

    def _emit(self, kind: str, text: str) -> None:
        if kind == "status":
            self.call_from_thread(self._set_status, text)
            return
        if kind == "hint":
            self.call_from_thread(self._add_hint, text)
            return
        if kind == "progress":
            self.call_from_thread(self._set_detail, text)
            return
        if kind == "stream_end":
            self.call_from_thread(self._end_stream, text == "aborted")
            return
        if kind.startswith("stream_"):
            self.call_from_thread(self._stream, kind[len("stream_"):], text)
            return
        if kind == "assistant":
            self.call_from_thread(self._write_markdown, text)
            return
        style = _KIND_STYLES.get(kind, "")
        self.call_from_thread(self._write, Text(("  " if kind == "result" else "") + text, style=style))

    def _approve(self, name: str, args: dict) -> bool:
        if not self.supervision:
            return True  # the command guard has already run; blocked commands never reach here
        self.call_from_thread(self._set_status, "Waiting for your approval")
        done, answer = threading.Event(), {"ok": False}

        def cb(ok: bool | None) -> None:
            answer["ok"] = bool(ok)
            done.set()

        self.call_from_thread(self.push_screen, ApprovalScreen(name, args), cb)
        while not done.wait(0.25):
            if not self.is_running or self.agent.stopped():  # quit or stopped while the prompt was open: deny
                return False
        return answer["ok"]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def _reattach_tty() -> None:
    """When run as `curl ... | python3 -`, stdin is the (now empty) pipe.
    Point fd 0 back at the terminal so the TUI can read keystrokes."""
    if sys.stdin.isatty():
        return
    try:
        fd = os.open("/dev/tty", os.O_RDWR)
    except OSError:
        sys.exit("ai: needs an interactive terminal")
    os.dup2(fd, 0)
    os.close(fd)


def main() -> None:
    p = argparse.ArgumentParser(prog="ai", description="Local agentic coding assistant (LM Studio)")
    p.add_argument("prompt", nargs="*", help="optional prompt to submit immediately")
    p.add_argument("--url", default=os.getenv("AI_URL", "http://localhost:1234/v1"), help="LM Studio server URL")
    p.add_argument("--model", default=os.getenv("AI_MODEL"), help="model id (default: first loaded model)")
    p.add_argument("--root", default=".", help="workspace directory the agent may touch (default: cwd)")
    p.add_argument("--generation-timeout", type=float,
                   default=float(os.getenv("AI_GENERATION_TIMEOUT") or DEFAULT_GENERATION_TIMEOUT),
                   help="seconds one reply may run before it is stopped and 'continue' is sent to retry "
                        "(default: 600; 0 = no limit). Breaks loops where the model repeats itself forever")
    p.add_argument("--max-continues", type=int, default=int(os.getenv("AI_MAX_CONTINUES") or 5),
                   help="how many generation timeouts in a row to retry before giving up (default: 5; 0 = no limit)")
    p.add_argument("--timeout", type=float, default=float(os.getenv("AI_TIMEOUT") or DEFAULT_TIMEOUT),
                   help="seconds to wait for LM Studio to send *any* data before giving up (default: 600). "
                        "Replies are streamed, so slow generation alone never trips this")
    p.add_argument("--session", metavar="FILE", default=os.getenv("AI_SESSION"),
                   help=f"session log to save to and resume from (default: {SESSION_FILE} in the workspace root)")
    p.add_argument("--new", action="store_true",
                   help="start a new session; an existing session log is moved aside, not deleted")
    p.add_argument("--no-session", action="store_true", help="don't save or resume a session")
    a = p.parse_args()

    _reattach_tty()
    toolbox = Toolbox(Path(a.root))
    session = None
    if not a.no_session:
        session = SessionLog(Path(a.session).expanduser() if a.session else toolbox.root / SESSION_FILE, toolbox.root)
        toolbox.protect(session.path)
    llm = LMStudio(a.url, a.model, timeout=a.timeout, generation_timeout=a.generation_timeout)
    agent = Agent(llm, toolbox, max_continues=a.max_continues, session=session)
    resume = None
    try:
        if session and a.new:
            session.archive()
        elif session:
            resume = agent.resume()
    except (SessionLocked, SessionVersionError) as e:
        sys.exit(f"ai: {e}. Close the other ai, or use --session FILE or --no-session.")
    try:
        AiApp(agent, initial_prompt=" ".join(a.prompt).strip() or None, resume=resume).run()
    finally:
        if session:
            session.close()


if __name__ == "__main__":
    main()