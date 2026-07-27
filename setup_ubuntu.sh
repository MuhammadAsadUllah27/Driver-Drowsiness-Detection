#!/usr/bin/env bash
# ============================================================
#  Driver Drowsiness Detection — Ubuntu Setup Script
#  Tested on Ubuntu 20.04, 22.04, 24.04
# ============================================================
set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Driver Drowsiness Detection — Ubuntu Setup        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 \
    v4l-utils \
    ffmpeg \
    libportaudio2 \
    --no-install-recommends
echo "  ✓ System packages installed."

# ── 2. Python virtual environment ─────────────────────────────────────────────
echo ""
echo "[2/6] Creating Python virtual environment at ./venv ..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools -q
echo "  ✓ Virtual environment ready."

# ── 3. PyTorch ────────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Installing PyTorch..."

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    echo "  GPU detected — CUDA ${CUDA_VERSION}"
    if [[ "$CUDA_VERSION" == 12* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    elif [[ "$CUDA_VERSION" == 11* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    else
        pip install torch torchvision torchaudio -q
    fi
else
    echo "  No GPU detected — installing CPU-only PyTorch."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi
echo "  ✓ PyTorch installed."

# ── 4. Python requirements ────────────────────────────────────────────────────
echo ""
echo "[4/6] Installing Python requirements..."
pip install -r requirements.txt -q
echo "  ✓ Requirements installed."

# ── 5. Camera permission ──────────────────────────────────────────────────────
echo ""
echo "[5/6] Adding user to 'video' group for camera access..."
sudo usermod -aG video "$USER"
echo "  ✓ Done (re-login required to take effect)."

# ── 6. Check camera ───────────────────────────────────────────────────────────
echo ""
echo "[6/6] Checking cameras..."
if ls /dev/video* 2>/dev/null | head -5; then
    echo "  ✓ Camera device(s) found."
else
    echo "  ⚠  No /dev/video* found. Connect a USB camera or enable built-in webcam."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Setup complete!                                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Activate environment any time with:"
echo "    source venv/bin/activate"
echo ""
echo "  Next steps:"
echo "    1. Prepare your dataset:"
echo "       Create the following folder structure manually (or with your own tool):"
echo "         dataset/train/awake   dataset/train/drowsy"
echo "         dataset/val/awake     dataset/val/drowsy"
echo ""
echo "    2. Train the model:"
echo "       python train.py"
echo ""
echo "    3. Run real-time detection:"
echo "       python detect.py"
echo ""
echo "    ⚡ Quick test (no training needed):"
echo "       python detect.py --geometry-only"
echo ""
echo "  Notes:"
echo "    - Alert clips are disabled by default (no files will be saved)."
echo "      To re-enable saving alert clips, set \`save_alert_clips = True\` in"
echo "      \`config/config.py\` and ensure the system has the required codecs."
echo ""
