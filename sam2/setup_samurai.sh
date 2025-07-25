#!/usr/bin/env bash
#-----------------------------------------------------------------------------
# setup_samurai.sh – bootstrap script for SAMURAI + checkpoints
#
# Place this file in <repo‑root>/sam2/  (yes: same folder you `import sam2` from)
#
# USAGE
#   cd sam2          # <- the script’s directory
#   bash setup_samurai.sh [large|base_plus|small|tiny]
#
# NOTES
# • First arg (optional) chooses which SAMURAI checkpoint to download. Default = large.
# • The script clones *yangchris11/samurai* (not just facebookresearch/sam2).
#   That repo already contains a vendor‑copied `sam2/` Python package that
#   exposes `build_sam2_video_predictor`, so your existing imports keep working.
# • Requires: git, curl (or wget), and—if you plan to use GPU—the NVIDIA driver
#   that matches the CUDA wheels you install (≥ 535 for CUDA 12.1).
#-----------------------------------------------------------------------------
set -euo pipefail

###############################################################################
# 0. Resolve paths
###############################################################################
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_DIR" )"
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

###############################################################################
# 1. Clone / update SAMURAI (which includes SAM‑2) into *this* sam2/ folder
###############################################################################
SAMURAI_REMOTE="https://github.com/yangchris11/samurai.git"

if [[ -d "${SCRIPT_DIR}/.git" ]]; then
  echo "➤ Updating existing SAMURAI clone inside sam2/ ..."
  git -C "$SCRIPT_DIR" pull --ff-only
else
  echo "➤ Cloning SAMURAI repository into sam2/ ..."
  rm -rf "${SCRIPT_DIR:?}/"*  # remove placeholder files, keep the script itself
  git clone --depth 1 "$SAMURAI_REMOTE" "$SCRIPT_DIR"
  echo "✔ SAMURAI source ready."
fi

###############################################################################
# 2. Download the requested checkpoint
###############################################################################
MODEL_SIZE="${1:-large}"   # default = large
declare -A URLS=(
  [large]="https://dl.fbaipublicfiles.com/samurai/models/sam2.1_hiera_large.pt"
  [base_plus]="https://dl.fbaipublicfiles.com/samurai/models/sam2.1_hiera_b+.pt"
  [small]="https://dl.fbaipublicfiles.com/samurai/models/sam2.1_hiera_s.pt"
  [tiny]="https://dl.fbaipublicfiles.com/samurai/models/sam2.1_hiera_t.pt"
)

if [[ -z "${URLS[$MODEL_SIZE]:-}" ]]; then
  echo "✖ Unknown model size \"$MODEL_SIZE\". Choose: large | base_plus | small | tiny"
  exit 1
fi

FILE="sam2.1_hiera_${MODEL_SIZE}.pt"
DEST="${CHECKPOINT_DIR}/${FILE}"

if [[ -f "$DEST" ]]; then
  echo "✔ Checkpoint $FILE already present – skipping download."
else
  echo "➤ Downloading SAMURAI checkpoint ($MODEL_SIZE) ..."
  curl -L "${URLS[$MODEL_SIZE]}" -o "$DEST"
fi

echo -e "\n✅  Setup complete."
echo "   • SAMURAI + SAM‑2 source lives in: sam2/"
echo "   • Model weights           in: checkpoints/${FILE}"
echo "   • You can now run:  python -m sam2_masking --video <file> --checkpoint checkpoints/${FILE}"
