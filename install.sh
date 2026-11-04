#!/bin/bash
set -euo pipefail

DATA_DIR=~/.local/share/whisper-dictation
VENV="$DATA_DIR/venv"
MODEL_DIR="$DATA_DIR/models"
MODEL_FILE="$MODEL_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/lmstudio-community/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
PARAKEET_DIR="$MODEL_DIR/parakeet-tdt-0.6b-v2"
PARAKEET_REPO="istupakov/parakeet-tdt-0.6b-v2-onnx"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== whisper-dictation installer ==="
echo

# Check system dependencies
MISSING=()
command -v xclip   >/dev/null || MISSING+=(xclip)
command -v xdotool >/dev/null || MISSING+=(xdotool)
command -v notify-send >/dev/null || MISSING+=(libnotify)
pkg-config --exists portaudio-2.0 2>/dev/null || MISSING+=(portaudio)

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Missing system dependencies: ${MISSING[*]}"
    echo "Install them with your package manager, e.g.:"
    echo "  sudo pacman -S ${MISSING[*]}    # Arch"
    echo "  sudo apt install ${MISSING[*]}  # Debian/Ubuntu"
    echo
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy] ]] || exit 1
fi

# Create directories
echo "Creating directories..."
mkdir -p "$MODEL_DIR"
mkdir -p ~/.local/bin

# Create venv
if [ ! -d "$VENV" ]; then
    echo "Creating Python venv..."
    python3 -m venv "$VENV"
else
    echo "Venv already exists, skipping creation."
fi

echo "Installing Python dependencies..."
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

# Install llama-cpp-python with CUDA support
echo "Installing llama-cpp-python with CUDA backend..."
CMAKE_ARGS="-DGGML_CUDA=on" "$VENV/bin/pip" install llama-cpp-python==0.3.16 --force-reinstall --no-cache-dir

# Install NVIDIA CUDA pip packages for LD_LIBRARY_PATH
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

# Download GGUF model
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading Qwen 2.5 7B Instruct GGUF model (~4.4 GB)..."
    curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
else
    echo "Model already downloaded, skipping."
fi

# Install files
echo "Installing dictation.py..."
cp "$SCRIPT_DIR/dictation.py" "$DATA_DIR/dictation.py"

echo "Installing launcher to ~/.local/bin/dictation..."
cp "$SCRIPT_DIR/dictation" ~/.local/bin/dictation
chmod +x ~/.local/bin/dictation

echo "Installing profile switcher to ~/.local/bin/dictation-profile..."
cp "$SCRIPT_DIR/dictation-profile" ~/.local/bin/dictation-profile
chmod +x ~/.local/bin/dictation-profile

# Install systemd service
echo "Installing systemd user service..."
mkdir -p ~/.config/systemd/user
cp "$SCRIPT_DIR/whisper-dictation.service" ~/.config/systemd/user/
systemctl --user daemon-reload

echo
echo "=== Installation complete ==="
echo
echo "Make sure ~/.local/bin is in your PATH."
echo
echo "To start and enable the service:"
echo
echo "  systemctl --user enable --now whisper-dictation.service"
echo
echo "Or run directly: dictation"
echo "Hold the right Super key to record, release to transcribe and paste."
echo "Or double-tap it to record hands-free, and double-tap again to transcribe."
echo
echo "To use Parakeet instead of Whisper: dictation-profile parakeet"
