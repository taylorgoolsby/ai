# ai

A small agentic coding assistant for your terminal, powered by a local model
served by LM Studio. One file: `ai.py`.

## Setup
1. In LM Studio, load a tool-calling-capable model (Qwen Coder, Llama 3.1+, etc.).
2. Start the dev server: Developer tab -> Start Server (or `lms server start`). Default: http://localhost:1234.
3. Push `ai.py` to the root of the `main` branch of github.com/taylorgoolsby/ai, then add this to `~/.zshrc` / `~/.bashrc`:

```sh
ai () {
	curl -s https://raw.githubusercontent.com/taylorgoolsby/ai/master/ai.py | python3 - "$@"
}
ai, () { ai "$@"; }
```

First run installs `textual` and `httpx` into `~/.cache/ai/deps` (nothing global).

## Usage
```
ai                          # open the TUI
ai, what time is it         # open the TUI with the prompt already submitted
ai what time is it          # same thing
ai --root ~/my-project      # limit the agent to a directory (default: current dir)
```
Options: `--url`, `--model`, `--root` (or env `AI_URL`, `AI_MODEL`).

Quote prompts containing shell characters: `ai, "what's the time?"`.
In zsh, `alias 'ai,'='noglob ai'` (instead of the `ai,` function) avoids "no matches found" on a bare `?`.

Keys: Enter send, Ctrl+L new chat, Ctrl+Q quit. Writes, edits and shell commands ask for approval (y / n).
