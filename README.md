# whisper-dictation

GPU-accelerated push-to-talk dictation for Linux/X11. Hold a key to record, release to transcribe and paste.

Uses [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text and [LLaMA CPP](https://github.com/abetlen/llama-cpp-python) with a [Qwen 2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model to clean up punctuation, capitalization, and formatting.

## How it works

1. Hold the trigger key (default: right Super) to record audio
2. Release to transcribe with Faster Whisper (GPU-accelerated)
3. The raw transcript is passed through a local LLaMA model that fixes punctuation, capitalization, and self-corrections
4. The cleaned text is copied to the clipboard and pasted into the focused window via `xdotool`

The formatting model handles things like:
- Fixing punctuation and capitalization
- Resolving self-corrections ("eggs milk no wait not milk butter" → "eggs, butter")
- Detecting sentence restarts ("I think we should deploy to I think we should deploy to production" → "I think we should deploy to production.")
- Handling retractions ("commit and push well not push the changes" → "commit the changes")
- Converting dictated structure ("new paragraph", "bullet point", "number one") into actual formatting
- Detecting implicit lists from natural speech patterns

## System requirements

- Linux with X11 (Wayland not supported — uses `xdotool` and `xclip`)
- NVIDIA GPU with CUDA support
- Python 3.10+

## System dependencies

Install these with your package manager:

```bash
# Arch
sudo pacman -S xclip xdotool libnotify portaudio

# Debian/Ubuntu
sudo apt install xclip xdotool libnotify-bin portaudio19-dev
```

## Installation

```bash
git clone https://github.com/tvdavies/whisper-dictation.git
cd whisper-dictation
./install.sh
```

The install script will:
- Create a Python venv at `~/.local/share/whisper-dictation/venv/`
- Install Python dependencies including CUDA-enabled `llama-cpp-python`
- Download the Qwen 2.5 7B Instruct GGUF model (~4.4 GB)
- Install the launcher to `~/.local/bin/dictation`

## Usage

### systemd service (recommended)

The install script installs a systemd user service. Enable it to start automatically with your graphical session:

```bash
systemctl --user enable --now whisper-dictation.service
```

Manage it with:

```bash
systemctl --user start whisper-dictation    # start
systemctl --user stop whisper-dictation     # stop
systemctl --user restart whisper-dictation  # restart after config changes
systemctl --user status whisper-dictation   # check status
```

### Run directly

```bash
dictation
```

### i3 keybinding

Alternatively, add to `~/.config/i3/config`:

```
bindsym $mod+d exec --no-startup-id ~/.local/bin/dictation
```

### CLI flags

```
--model MODEL       Whisper model (default: large-v3-turbo)
                    Options: tiny, base, small, medium, large-v3, large-v3-turbo
--language LANG     Language code or 'auto' (default: en)
--device DEVICE     cuda or cpu (default: cuda)
--compute-type TYPE float16, int8_float16, int8, float32 (default: float16)
--key KEY           Push-to-talk key (default: super_r)
                    Options: super_r, caps_lock, f8, pause, etc.
--no-format         Disable LLM formatting (raw Whisper output)
--format-model PATH Path to a custom GGUF model for formatting
```

### Examples

```bash
dictation                          # defaults
dictation --model small.en         # lighter model, English-only
dictation --key caps_lock          # use Caps Lock as trigger
dictation --language auto          # auto-detect language
dictation --no-format              # skip LLM cleanup, raw Whisper output
```

### Logs

When running via systemd:

```bash
journalctl --user -u whisper-dictation -f
```

When running directly, output is logged to `~/.local/share/whisper-dictation/dictation.log`.
