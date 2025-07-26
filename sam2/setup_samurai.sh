#!/usr/bin/env bash
set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SAM_REPO_DIR="$SCRIPT_DIR/sam2"

# Clone SAMURAI repository if not present
if [ ! -d "$SAM_REPO_DIR" ]; then
    git clone https://github.com/yangchris11/samurai "$SAM_REPO_DIR"
fi

# Install SAMURAI in editable mode with notebooks extras
pip install -e "$SAM_REPO_DIR"
pip install -e "$SAM_REPO_DIR"\[notebooks\]

# Download checkpoints
bash "$SAM_REPO_DIR/checkpoints/download_ckpts.sh"

# Copy chosen checkpoint to top-level checkpoints directory
CKPT_NAME="${1:-sam2.1_hiera_large.pt}"
mkdir -p "$ROOT_DIR/checkpoints"
cp "$SAM_REPO_DIR/checkpoints/$CKPT_NAME" "$ROOT_DIR/checkpoints/"

echo "Checkpoint copied to $ROOT_DIR/checkpoints/$CKPT_NAME"
