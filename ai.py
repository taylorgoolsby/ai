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
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------
# Bootstrap: make sure textual + httpx are importable without touching the
# system Python (pip --target into a private cache dir).
# --------------------------------------------------------------------------
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
from rich.markdown import Markdown  # noqa: E402
from rich.text import Text  # noqa: E402
from textual import work  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Horizontal, Vertical  # noqa: E402
from textual.screen import ModalScreen  # noqa: E402
from textual.widgets import Button, Footer, Header, Input, RichLog, Static  # noqa: E402


# --------------------------------------------------------------------------
# LM Studio client (OpenAI-compatible dev server)
# --------------------------------------------------------------------------
class LLMError(Exception):
    pass


class LMStudio:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http = httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0))

    def _request(self, method: str, path: str, **kw) -> dict:
        try:
            r = self.http.request(method, self.base_url + path, **kw)
            r.raise_for_status()
            return r.json()
        except httpx.ConnectError:
            raise LLMError(
                f"Can't reach LM Studio at {self.base_url}. "
                "Load a model and start the server (Developer tab -> Start Server, "
                "or `lms server start`)."
            )
        except httpx.HTTPStatusError as e:
            raise LLMError(f"LM Studio returned {e.response.status_code}: {e.response.text[:500]}")
        except httpx.HTTPError as e:
            raise LLMError(str(e))

    def list_models(self) -> list[str]:
        return [m["id"] for m in self._request("GET", "/models").get("data", [])]

    def ensure_model(self) -> str:
        if not self.model:
            models = [m for m in self.list_models() if "embed" not in m.lower()]
            if not models:
                raise LLMError("No model is loaded in LM Studio.")
            self.model = models[0]
        return self.model

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body = {"model": self.ensure_model(), "messages": messages, "temperature": 0.2}
        if tools:
            body["tools"] = tools
        data = self._request("POST", "/chat/completions", json=body)
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError):
            raise LLMError(f"Unexpected response: {str(data)[:500]}")


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
# Shell guardrails. These catch accidents (rm -rf /, sudo, cd / && rm *, force
# pushes...) before the approval prompt is even shown. They are a seatbelt, not
# a sandbox: `python -c "..."` can still do anything. Use a container for real
# isolation.
# --------------------------------------------------------------------------
ALWAYS_BLOCKED = {
    "sudo", "su", "doas", "pkexec", "mkswap", "fdisk", "sfdisk", "parted", "wipefs",
    "shutdown", "reboot", "halt", "poweroff", "init", "telinit", "mount", "umount",
    "chroot", "killall",
}
DESTRUCTIVE = {"rm", "rmdir", "mv", "shred", "truncate", "unlink"}
MUTATORS = DESTRUCTIVE | {"cp", "chmod", "chown", "chgrp", "ln", "install", "rsync", "tee", "touch", "mkdir"}
SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
INTERPRETERS = SHELLS | {"python", "python3", "perl", "ruby", "node"}
WRAPPERS = {"command", "builtin", "exec", "nohup", "time", "nice", "env", "stdbuf"}
KNOWN = ALWAYS_BLOCKED | MUTATORS | SHELLS | {"find", "xargs", "git", "dd", "kill", "cd", "pushd", "eval"}
OPERATOR_CHARS = set(";&|()`")
REDIRECTS = {">", ">>", "&>", "&>>", ">|"}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _tokenize(command: str) -> list[str]:
    lex = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|()`<>")
    lex.whitespace_split = True
    return list(lex)


def _segments(tokens: list[str]) -> list[tuple[str, list[str]]]:
    segs: list[tuple[str, list[str]]] = []
    cur: list[str] = []
    sep = ""
    for t in tokens:
        if t and set(t) <= OPERATOR_CHARS:
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
    while i < len(words) and ASSIGN.match(words[i]):
        i += 1
    if i < len(words) and os.path.basename(words[i]) in WRAPPERS:
        for j in range(i + 1, len(words)):
            if os.path.basename(words[j]) in KNOWN:
                return words[j:]
        return words[i + 1:]
    return words[i:]


def _resolve_in_root(arg: str, root: Path) -> tuple[Path | None, str | None]:
    if arg.startswith("~") or "$" in arg or "`" in arg:
        return None, f"can't verify path '{arg}' (uses ~ or a variable); use a literal path inside the workspace"
    p = (root / arg).resolve()
    if not p.is_relative_to(root):
        return None, f"path '{arg}' is outside the workspace"
    return p, None


def _check_words(sep: str, prev_cmd: str, words: list[str], root: Path, depth: int) -> str | None:
    # redirections must stay inside the workspace (or /dev/null)
    for i, t in enumerate(words):
        if t in REDIRECTS and i + 1 < len(words) and words[i + 1] != "/dev/null":
            _, err = _resolve_in_root(words[i + 1], root)
            if err:
                return f"redirect target: {err}"

    words = _command_start(words)
    if not words:
        return None
    cmd, args = os.path.basename(words[0]), words[1:]
    flagless = [a for a in args if not a.startswith("-")]

    if cmd in ALWAYS_BLOCKED or cmd.startswith("mkfs"):
        return f"'{cmd}' is not allowed"
    if sep == "|" and prev_cmd in {"curl", "wget"} and cmd in INTERPRETERS:
        return "piping a download into a shell or interpreter is not allowed"
    if cmd in SHELLS:
        for i, a in enumerate(args):
            if a.startswith("-") and not a.startswith("--") and "c" in a and i + 1 < len(args):
                return shell_danger(args[i + 1], root, depth + 1)
    if cmd == "eval":
        return shell_danger(" ".join(args), root, depth + 1)
    if cmd == "xargs":
        if any(os.path.basename(a) in (MUTATORS | ALWAYS_BLOCKED | SHELLS) for a in args):
            return "xargs with file-modifying commands is not allowed (paths can't be verified); name the paths or use find -delete"
    if cmd in {"cd", "pushd"}:
        if len(flagless) != 1 or flagless[0] == "-":
            return "cd must name one directory inside the workspace"
        _, err = _resolve_in_root(flagless[0], root)
        if err:
            return err
    if cmd in MUTATORS:
        targets = flagless[1:] if cmd in {"chmod", "chown", "chgrp"} else flagless
        for a in targets:
            p, err = _resolve_in_root(a, root)
            if err:
                return f"{cmd}: {err}"
            if cmd in DESTRUCTIVE and p in (root, root / ".git"):
                return f"{cmd}: refusing to modify the workspace root or .git"
    if cmd == "dd":
        for a in args:
            if a.startswith("of="):
                _, err = _resolve_in_root(a[3:], root)
                if err:
                    return f"dd: {err}"
    if cmd == "kill" and any(a in {"-1", "0"} for a in args):
        return "kill -1 / kill 0 would signal every process"
    if cmd == "find":
        starts = []
        for a in args:
            if a.startswith("-") or a in {"!", "("}:
                break
            starts.append(a)
        if "-delete" in args:
            for a in starts or ["."]:
                _, err = _resolve_in_root(a, root)
                if err:
                    return f"find -delete: {err}"
        for i, a in enumerate(args):
            if a in {"-exec", "-execdir", "-ok", "-okdir"}:
                r = _check_words("", "", args[i + 1:], root, depth)
                if r:
                    return r
    if cmd == "git":
        rest = list(args)
        while rest and rest[0] in {"-C", "-c"}:
            rest = rest[2:]
        sub = next((a for a in rest if not a.startswith("-")), "")
        if sub == "clean":
            return "git clean deletes untracked files; ask the user to run it"
        if sub == "reset" and "--hard" in rest:
            return "git reset --hard discards uncommitted work; ask the user to run it"
        if sub == "push" and any(a == "-f" or a.startswith("--force") for a in rest):
            return "force-pushing is not allowed"
    return None


def shell_danger(command: str, root: Path, depth: int = 0) -> str | None:
    """Return a reason if the command looks dangerous, else None."""
    if depth > 3:
        return "commands nested too deeply to verify"
    if re.search(r":\s*\(\s*\)\s*\{", command):
        return "fork bomb"
    try:
        tokens = _tokenize(command)
    except ValueError as e:
        return f"can't parse command ({e})"
    prev_cmd = ""
    for sep, words in _segments(tokens):
        reason = _check_words(sep, prev_cmd, words, root, depth)
        if reason:
            return reason
        start = _command_start(words)
        prev_cmd = os.path.basename(start[0]) if start else ""
    return None


class Toolbox:
    def __init__(self, root: Path):
        self.root = root.resolve()
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

    def read_file(self, path: str) -> str:
        return self._path(path).read_text(errors="replace")

    def write_file(self, path: str, content: str) -> str:
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = self._path(path)
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
        return shell_danger(command, self.root)

    def run_shell(self, command: str) -> str:
        reason = self._validate_shell(command)
        if reason:
            return f"Blocked: {reason}"
        proc = subprocess.Popen(
            command, shell=True, cwd=self.root, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # kill the whole process group, not just the shell
            proc.communicate()
            return "Error: command timed out after 120s and was killed"
        return f"exit code {proc.returncode}\n{out}{err}"

    def _build(self) -> list[Tool]:
        s = {"type": "string"}
        return [
            Tool("list_dir", "List files in a directory.", _params({"path": s}, []), self.list_dir),
            Tool("read_file", "Read a text file.", _params({"path": s}, ["path"]), self.read_file),
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
# Agent loop: model -> tool calls -> results -> model, until it answers
# --------------------------------------------------------------------------
SYSTEM = """You are ai, a careful software engineering agent working in {root}.
Use tools to inspect the project before changing it. Make small, focused edits
with edit_file when possible. After changes, verify them (run tests or the code).
Be concise. When the task is done, say what you changed."""

Emit = Callable[[str, str], None]       # (kind, text)
Approve = Callable[[str, dict], bool]   # (tool name, args) -> allowed?


class Agent:
    def __init__(self, llm: LMStudio, toolbox: Toolbox, max_steps: int = 25):
        self.llm, self.toolbox, self.max_steps = llm, toolbox, max_steps
        self.messages: list[dict] = []
        self.reset()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM.format(root=self.toolbox.root)}]

    def run(self, user_text: str, emit: Emit, approve: Approve) -> None:
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(self.max_steps):
            msg = self.llm.chat(self.messages, self.toolbox.schemas())
            calls = msg.get("tool_calls") or []
            entry: dict = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                entry["tool_calls"] = calls
            self.messages.append(entry)
            if entry["content"].strip():
                emit("assistant", entry["content"])
            if not calls:
                return
            for call in calls:
                self._run_call(call, emit, approve)
        emit("error", f"Stopped after {self.max_steps} steps.")

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
            elif tool and tool.dangerous and not approve(name, args):
                result = "User denied this action."
                emit("error", f"denied: {name}")
            else:
                result = self.toolbox.execute(name, args)
                emit("result", result[:400])
        self.messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})


# --------------------------------------------------------------------------
# Textual TUI
# --------------------------------------------------------------------------
class ApprovalScreen(ModalScreen[bool]):
    BINDINGS = [Binding("y", "allow", "Allow"), Binding("n,escape", "deny", "Deny")]
    DEFAULT_CSS = """
    ApprovalScreen { align: center middle; }
    #box { width: 80%; max-height: 80%; border: thick $warning; background: $surface; padding: 1 2; }
    #detail { height: auto; max-height: 20; overflow-y: auto; margin-bottom: 1; }
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


class AiApp(App):
    TITLE = "ai"
    BINDINGS = [Binding("ctrl+q", "quit", "Quit"), Binding("ctrl+l", "clear", "New chat")]
    CSS = "#log { height: 1fr; padding: 0 1; } #prompt { dock: bottom; }"

    def __init__(self, agent: Agent, initial_prompt: str | None = None):
        super().__init__()
        self.agent = agent
        self.initial_prompt = initial_prompt
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="log", wrap=True, markup=False, highlight=False)
        yield Input(placeholder="Ask ai to build something...  (/clear to reset)", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = str(self.agent.toolbox.root)
        self.query_one(Input).focus()
        self._write(Text(f"ai ready. Workspace: {self.agent.toolbox.root}", style="dim"))
        if self.initial_prompt:
            self.submit(self.initial_prompt)

    def _write(self, renderable) -> None:
        self.query_one(RichLog).write(renderable)

    def action_clear(self) -> None:
        self.agent.reset()
        self.query_one(RichLog).clear()
        self._write(Text("Conversation cleared.", style="dim"))

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
        self.run_agent(text)

    # runs in a worker thread; UI updates go through call_from_thread
    @work(thread=True, exclusive=True)
    def run_agent(self, text: str) -> None:
        try:
            self.agent.run(text, self._emit, self._approve)
        except LLMError as err:
            self._emit("error", str(err))
        except Exception as err:
            self._emit("error", f"{type(err).__name__}: {err}")
        finally:
            self.call_from_thread(self._done)

    def _done(self) -> None:
        self.busy = False
        self.sub_title = str(self.agent.toolbox.root)

    def _emit(self, kind: str, text: str) -> None:
        style = {"tool": "yellow", "result": "dim", "error": "bold red"}.get(kind, "")
        item = Markdown(text) if kind == "assistant" else Text(("  " if kind == "result" else "") + text, style=style)
        self.call_from_thread(self._write, item)

    def _approve(self, name: str, args: dict) -> bool:
        done, answer = threading.Event(), {"ok": False}

        def cb(ok: bool | None) -> None:
            answer["ok"] = bool(ok)
            done.set()

        self.call_from_thread(self.push_screen, ApprovalScreen(name, args), cb)
        done.wait()
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
    a = p.parse_args()

    _reattach_tty()
    agent = Agent(LMStudio(a.url, a.model), Toolbox(Path(a.root)))
    AiApp(agent, initial_prompt=" ".join(a.prompt).strip() or None).run()


if __name__ == "__main__":
    main()