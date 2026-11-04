# whisper-dictation

GPU-accelerated push-to-talk dictation for Linux, on Wayland or X11. Hold a key to record, release to transcribe and paste.

Speech-to-text runs locally with either NVIDIA [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (through [onnx-asr](https://github.com/istupakov/onnx-asr)) or [Faster Whisper](https://github.com/SYSTRAN/faster-whisper). An optional formatting pass sends the transcript to a local model served by [LM Studio](https://lmstudio.ai) to clean up punctuation, self-corrections and lists.

## How it works

1. Hold the trigger key (default: right Ctrl) to record audio, or double-tap it to record hands-free
2. Release it (or double-tap again) to transcribe with Parakeet or Whisper on the GPU
3. If LM Studio is running, the transcript goes through its formatting model; otherwise the raw transcript is used
4. The text is copied to the clipboard and pasted into the focused window with Ctrl+Shift+V (`wl-copy` + `wtype` on Wayland, `xclip` + `xdotool` on X11). It stays on the clipboard afterwards

The formatting pass handles things like:
- Fixing punctuation and capitalization
- Resolving self-corrections ("eggs milk no wait not milk butter" → "eggs, butter")
- Detecting sentence restarts ("I think we should deploy to I think we should deploy to production" → "I think we should deploy to production.")
- Handling retractions ("commit and push well not push the changes" → "commit the changes")
- Converting dictated structure ("new paragraph", "bullet point", "number one") into actual formatting
- Detecting implicit lists from natural speech patterns

## System requirements

- Linux on Wayland (with a compositor that supports the virtual keyboard protocol, e.g. Hyprland or Sway) or X11
- NVIDIA GPU with CUDA support (or `--device cpu`)
- Python 3.10+
- Membership of the `input` group, since the trigger key is read from `/dev/input`
- Optional: [LM Studio](https://lmstudio.ai) serving a small instruct model (e.g. Qwen3.5 4B) at `http://localhost:1234` for formatting

## System dependencies

```bash
# Arch, Wayland
sudo pacman -S wl-clipboard wtype libnotify glib2 portaudio
# Arch, X11
sudo pacman -S xclip xdotool libnotify glib2 portaudio

# Debian/Ubuntu, Wayland
sudo apt install wl-clipboard wtype libnotify-bin libglib2.0-bin portaudio19-dev
# Debian/Ubuntu, X11
sudo apt install xclip xdotool libnotify-bin libglib2.0-bin portaudio19-dev
```

## Installation

```bash
git clone https://github.com/tvdavies/whisper-dictation.git
cd whisper-dictation
./install.sh
```

The install script will:
- Create a Python venv at `~/.local/share/whisper-dictation/venv/` and install the Python dependencies and CUDA runtime libraries
- Swap the CPU `onnxruntime` for `onnxruntime-gpu` and download the Parakeet TDT 0.6B v2 ONNX model (~2.5 GB) to `~/.local/share/whisper-dictation/models/parakeet-tdt-0.6b-v2`
- Symlink `dictation.py`, the `dictation` launcher, `dictation-profile` and the systemd user service from the checkout, so keep the checkout and `git pull` to update
- Write the `parakeet` profile to `~/.config/whisper-dictation/env` if there's no profile yet

Re-running it over an existing install is safe. Whisper models download on first use.

## Usage

Hold the trigger key to record and release it to transcribe and paste. Or double-tap it to record hands-free, then double-tap again to transcribe. A single quick tap is ignored, and a press combined with another key (such as Right Ctrl+C) never counts as a tap.

### Profiles

`dictation-profile` rewrites `~/.config/whisper-dictation/env` and restarts the service:

```bash
dictation-profile parakeet  # GPU Parakeet TDT 0.6B v2 + LM Studio formatting
dictation-profile gpu       # GPU Whisper large-v3-turbo + LM Studio formatting
dictation-profile cpu       # CPU Whisper tiny.en, no formatting
dictation-profile status
```

### systemd service (recommended)

Enable the service to start automatically with your session:

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
dictation --engine parakeet --device cuda
```

Running `dictation` from a terminal stops the service first, so the two don't fight over the key.

### CLI flags

```
--engine ENGINE     whisper or parakeet (default: whisper). parakeet ignores
                    --model and --compute-type
--parakeet-model DIR
                    Parakeet ONNX model directory (default:
                    ~/.local/share/whisper-dictation/models/parakeet-tdt-0.6b-v2)
--model MODEL       Whisper model (default: tiny.en)
                    Options: tiny, base, small, medium, large-v3, large-v3-turbo,
                    distil-large-v3; add .en for English-only variants
--language LANG     Whisper language code or 'auto' (default: en)
--device DEVICE     cuda or cpu (default: cpu)
--compute-type TYPE Whisper compute type: float16, int8_float16, int8, float32
                    (default: int8)
--key KEY           Push-to-talk key (default: super_r, which reads the right
                    Ctrl key). Options: super_r, caps_lock, f8, pause, etc.
--no-format         Disable the LM Studio formatting pass (raw transcript)
--lmstudio-url URL  LM Studio OpenAI-compatible base URL
                    (default: http://localhost:1234/v1)
--lmstudio-model ID Formatting model identifier (default: qwen/qwen3-4b-2507)
--lmstudio-reasoning-effort EFFORT
                    reasoning_effort sent with format requests; 'none' turns
                    off thinking on models like Qwen3.5 (default: omitted)
--sal-voice CMD     Run `CMD hold press` / `CMD hold release` around each
                    recording so a live Sal voice call is muted while you
                    dictate. 'auto' (default) finds sal-voice on PATH or in
                    ~/.local/bin; 'none' disables
```

### Examples

```bash
dictation --engine parakeet --device cuda   # Parakeet on the GPU
dictation --model large-v3-turbo --device cuda --compute-type float16
dictation --key caps_lock                   # use Caps Lock as trigger
dictation --language auto                   # auto-detect language (Whisper)
dictation --no-format                       # skip formatting, raw transcript
```

### Logs

When running via systemd:

```bash
journalctl --user -u whisper-dictation -f
```

When running directly, output goes to the terminal.
