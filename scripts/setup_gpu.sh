#!/usr/bin/env bash
# One-time setup of the GPU stages on a Linux machine with an NVIDIA GPU (Colab, Kaggle, a lab box).
#
# The three third-party tools pin incompatible packages (two different libraries are both called
# "pycolmap", and gsplat's examples want numpy<2), so each gets its own virtual environment that
# reuses the system's PyTorch:
#
#   envs/draft   MapAnything  feed-forward metric poses + dense points   (Apache-2.0 code and weights)
#   envs/sfm     pycolmap     bundle-adjusted poses, plus this repo's CPU tools (BSD-3)
#   envs/train   gsplat       Gaussian-splat training                      (Apache-2.0)
#
# plus splat-transform (MIT) from npm for compression.
#
# NOTE: written from the upstream READMEs; the authoring environment had no GPU, so this script
# has not been executed end to end. If a step fails, the upstream install notes are linked below.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PYTHON:-python3}
mkdir -p third_party envs

$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to PyTorch'; print('torch', torch.__version__, 'cuda ok')"

mkenv () {  # mkenv <name>: venv that can see the system torch
  if [ ! -x "envs/$1/bin/python" ]; then
    $PY -m venv --system-site-packages "envs/$1" 2>/dev/null || { $PY -m pip install -q virtualenv; $PY -m virtualenv --system-site-packages "envs/$1"; }
  fi
}

# ---- 1. CPU tools + COLMAP bindings --------------------------------------------------------------
mkenv sfm
envs/sfm/bin/pip install -q -r requirements.txt "pycolmap==4.2.1"

# ---- 2. MapAnything: https://github.com/facebookresearch/map-anything#installation ----------------
mkenv draft
[ -d third_party/map-anything ] || git clone --depth 1 https://github.com/facebookresearch/map-anything.git third_party/map-anything
envs/draft/bin/pip install -q -e "third_party/map-anything[colmap]"

# ---- 3. gsplat: https://github.com/nerfstudio-project/gsplat#installation --------------------------
mkenv train
[ -d third_party/gsplat ] || git clone --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git third_party/gsplat
envs/train/bin/pip install -q "gsplat==1.5.3"
envs/train/bin/pip install -q -r third_party/gsplat/examples/requirements.txt

# ---- 4. splat-transform: https://github.com/playcanvas/splat-transform -----------------------------
command -v node >/dev/null || { echo "Node.js 18+ is needed for splat-transform and the viewer"; exit 1; }
[ -d third_party/node_modules/@playcanvas/splat-transform ] || npm install --prefix third_party @playcanvas/splat-transform@3

echo
echo "Setup done. Next: bash scripts/run_pipeline.sh <video.mp4> [work_dir]"
