#!/bin/bash
set -euo pipefail

DATA_DIR=~/.local/share/whisper-dictation
VENV="$DATA_DIR/venv"
MODEL_DIR="$DATA_DIR/models"
PARAKEET_DIR="$MODEL_DIR/parakeet-tdt-0.6b-v2"
PARAKEET_REPO="istupakov/parakeet-tdt-0.6b-v2-onnx"
CONFIG_DIR=~/.config/whisper-dictation
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== whisper-dictation installer ==="
echo

# Check system dependencies. Pasting uses wl-copy + wtype on Wayland and
# xclip + xdotool on X11; gdbus dismisses the recording notification.
MISSING=()
need() { command -v "$1" >/dev/null || MISSING+=("$2"); }
if [ -n "${WAYLAND_DISPLAY:-}" ]; then
    need wl-copy wl-clipboard
    need wtype wtype
else
    need xclip xclip
    need xdotool xdotool
fi
need notify-send libnotify
need gdbus glib2
pkg-config --exists portaudio-2.0 2>/dev/null || MISSING+=(portaudio)

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Missing system dependencies (Arch package names): ${MISSING[*]}"
    echo "  sudo pacman -S ${MISSING[*]}"
    echo "On Debian/Ubuntu, libnotify is libnotify-bin, glib2 is libglib2.0-bin"
    echo "and portaudio is portaudio19-dev."
    echo
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy] ]] || exit 1
fi

# The trigger key is read straight from /dev/input, which needs the input group.
if ! id -nG | tr ' ' '\n' | grep -qx input; then
    echo "Warning: you are not in the 'input' group, so the trigger key can't be read."
    echo "  sudo usermod -aG input \"$USER\"   # then log out and back in"
    echo
fi

echo "Creating directories..."
mkdir -p "$MODEL_DIR" "$CONFIG_DIR" ~/.local/bin ~/.config/systemd/user

if [ ! -d "$VENV" ]; then
    echo "Creating Python venv..."
    python3 -m venv "$VENV"
else
    echo "Venv already exists, skipping creation."
fi

echo "Installing Python dependencies..."
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

# CUDA libraries for both engines; the launcher puts them on LD_LIBRARY_PATH.
echo "Installing NVIDIA CUDA runtime libraries..."
"$VENV/bin/pip" install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12

# Parakeet engine (--engine parakeet) runs on onnxruntime-gpu. faster-whisper
# pulls in the CPU onnxruntime, which provides the same module and shadows the
# GPU build, so remove it and reinstall the GPU build over the top.
echo "Switching onnxruntime to the CUDA build for Parakeet..."
ONNXRUNTIME_GPU=$(grep -E '^onnxruntime-gpu==' "$SCRIPT_DIR/requirements.txt")
"$VENV/bin/pip" uninstall -y onnxruntime || true
"$VENV/bin/pip" install --force-reinstall --no-deps "$ONNXRUNTIME_GPU"

# Download the Parakeet model into a plain directory. Not the Hugging Face
# cache: its weights are symlinks into a blobs dir, and onnxruntime refuses
# external weights that resolve outside the model directory.
if [ ! -f "$PARAKEET_DIR/encoder-model.onnx.data" ]; then
    echo "Downloading Parakeet TDT 0.6B v2 ONNX model (~2.5 GB)..."
    "$VENV/bin/python3" - "$PARAKEET_REPO" "$PARAKEET_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

# The int8 files are a quantised variant the engine doesn't load.
snapshot_download(sys.argv[1], local_dir=sys.argv[2], ignore_patterns=["*.int8.onnx"])
PY
else
    echo "Parakeet model already downloaded, skipping."
fi

# Link rather than copy, so a `git pull` in this checkout updates the install
# and re-running this script over an existing install is safe. ln -sfn also
# replaces regular files left by older versions of this script.
echo "Linking dictation.py, launcher, profile switcher and systemd service..."
ln -sfn "$SCRIPT_DIR/dictation.py" "$DATA_DIR/dictation.py"
ln -sfn "$SCRIPT_DIR/dictation" ~/.local/bin/dictation
ln -sfn "$SCRIPT_DIR/dictation-profile" ~/.local/bin/dictation-profile
ln -sfn "$SCRIPT_DIR/whisper-dictation.service" ~/.config/systemd/user/whisper-dictation.service
chmod +x "$SCRIPT_DIR/dictation" "$SCRIPT_DIR/dictation-profile"
systemctl --user daemon-reload

# The service reads its arguments from the env file; start on Parakeet.
if [ ! -f "$CONFIG_DIR/env" ]; then
    echo "Writing the default (parakeet) profile to $CONFIG_DIR/env..."
    DICTATION_PROFILE_NO_RESTART=1 "$SCRIPT_DIR/dictation-profile" parakeet
else
    echo "Keeping the existing profile in $CONFIG_DIR/env."
fi

echo
echo "=== Installation complete ==="
echo
echo "Keep this checkout: the installed files are symlinks into it."
echo "Make sure ~/.local/bin is in your PATH."
echo
echo "To start and enable the service:"
echo
echo "  systemctl --user enable --now whisper-dictation.service"
echo
echo "Or run directly: dictation \$(grep '^DICTATION_ARGS=' $CONFIG_DIR/env | cut -d= -f2-)"
echo
echo "Hold the right Ctrl key to record, release to transcribe and paste."
echo "Or double-tap it to record hands-free, and double-tap again to transcribe."
echo "Switch engines with: dictation-profile {parakeet|gpu|cpu}"
echo
echo "Formatting (punctuation, self-corrections, lists) uses a model served by"
echo "LM Studio at http://localhost:1234. Without it, raw transcripts are pasted."
