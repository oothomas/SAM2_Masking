#!/usr/bin/env bash
#-----------------------------------------------------------------------------
# setup_samurai.sh  –  one-shot bootstrapper for SAMURAI + checkpoints
#
# Works even when sam2/ already exists and isn’t a git repo.
# Place this script inside <repo-root>/sam2/ and run:
#
#   cd sam2
#   bash setup_samurai.sh [large|base_plus|small|tiny]
#
# Requirements: git, curl (or wget). CUDA ≥ 12.1 driver if you’ll use GPU wheels.
#-----------------------------------------------------------------------------
set -euo pipefail

###############################################################################
# 0. Resolve paths
###############################################################################
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_DIR" )"
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

###############################################################################
# 1. Sync SAMURAI (includes SAM-2) into *this* sam2/ folder
###############################################################################
SAMURAI_REMOTE="https://github.com/yangchris11/samurai.git"
SCRIPT_NAME="$( basename "${BASH_SOURCE[0]}" )"

sync_repo () {
  if [[ -d "${SCRIPT_DIR}/.git" ]]; then
    echo "➤ Updating existing SAMURAI git repo ..."
    git -C "$SCRIPT_DIR" pull --ff-only
  else
    echo "➤ sam2/ isn’t a git repo – cloning fresh source ..."
    TMP_DIR="$(mktemp -d)"
    git clone --depth 1 "$SAMURAI_REMOTE" "$TMP_DIR"
    echo "➤ Syncing source into sam2/ (overwriting everything except this script) ..."
    rsync -a --delete \
          --exclude "$SCRIPT_NAME" \
          "$TMP_DIR/" "$SCRIPT_DIR/"
    rm -rf "$TMP_DIR"
    echo "✔ SAMURAI source ready."
  fi
}
sync_repo

###############################################################################
# 2. Download the requested checkpoint
###############################################################################
MODEL_SIZE="${1:-large}"   # default = large
declare -A URLS=(
  [large]="https://dl.fbaipublicfiles.com/samurai/models/sam2.1_hiera_large.pt"
  [base_plus]="https://dl.fbaipublicfiles.com/samurai/models/sam2.1_hiera_b+.pt"
  [small]="https://dl.fbaipublicfiles.com/samurai/models/sam2.1_hiera_s.pt"
  [tiny]="https://dl.fbaipublicfiles.com/samurai/models/sam2.1_hiera_t.pt"
)

[[ -n "${URLS[$MODEL_SIZE]:-}" ]] || {
  echo "✖ Unknown model size \"$MODEL_SIZE\". Choose: large | base_plus | small | tiny"
  exit 1
}

FILE="sam2.1_hiera_${MODEL_SIZE}.pt"
DEST="${CHECKPOINT_DIR}/${FILE}"

if [[ -f "$DEST" ]]; then
  echo "✔ Checkpoint $FILE already present – skipping download."
else
  echo "➤ Downloading SAMURAI checkpoint ($MODEL_SIZE) ..."
  curl -L "${URLS[$MODEL_SIZE]}" -o "$DEST"
fi

###############################################################################
# 3. Done
###############################################################################
cat <<EOF

✅  Setup complete.
   • SAMURAI + SAM-2 source: sam2/
   • Model weights        : checkpoints/${FILE}

Run:
   python -m sam2_masking --video <your_video.mp4> --checkpoint checkpoints/${FILE}

EOF
