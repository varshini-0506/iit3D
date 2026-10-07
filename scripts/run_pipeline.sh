#!/usr/bin/env bash
# NammaTwin, video to walkable web twin.
#
#   bash scripts/run_pipeline.sh walkthrough.mp4 work            # first pass: stops at the recapture ticket
#   bash scripts/run_pipeline.sh --append topup.mp4 work         # add recapture clips, re-audit
#   bash scripts/run_pipeline.sh --finish work                   # final poses, training, packaging, report
#
# Settings (environment variables):
#   MARKER_SIZE=0.150   printed edge of the ArUco marker in metres (omit if no marker was used)
#   CAMERA_HEIGHT=1.45  fallback scale when there is no marker and you do not trust the draft's metric scale
#   FINAL_POSES=colmap  colmap (bundle-adjusted, default) | draft (feed-forward poses + gsplat pose refinement)
#   MAX_GAUSSIANS=1500000   budget for the trained splat; this fixes download size and frame rate
#   STEPS=30000         training steps
#   TITLE="My home"     shown in the viewer
#   SCENE_DIR=viewer/public/scene   where the finished scene is published
#   FORCE=1             carry on past a FAIL from the linter or a partial COLMAP model
#   SPLAT_TRANSFORM_ARGS="-g cpu"   compress on the CPU when the machine has no WebGPU adapter
#
# Stages 2, 5 and 6 need a GPU and the environments made by scripts/setup_gpu.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY_CPU=${PY_CPU:-$( [ -x "$ROOT/envs/sfm/bin/python" ] && echo "$ROOT/envs/sfm/bin/python" || echo python3 )}
PY_DRAFT=${PY_DRAFT:-$ROOT/envs/draft/bin/python}
PY_TRAIN=${PY_TRAIN:-$ROOT/envs/train/bin/python}
SPLAT_TRANSFORM=${SPLAT_TRANSFORM:-$ROOT/third_party/node_modules/.bin/splat-transform}
MAPANYTHING=${MAPANYTHING:-$ROOT/third_party/map-anything}
GSPLAT=${GSPLAT:-$ROOT/third_party/gsplat}
FINAL_POSES=${FINAL_POSES:-colmap}
MAX_GAUSSIANS=${MAX_GAUSSIANS:-1500000}
STEPS=${STEPS:-30000}
TITLE=${TITLE:-NammaTwin scene}
SCENE_DIR=${SCENE_DIR:-$ROOT/viewer/public/scene}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
nt () { "$PY_CPU" -m nammatwin "$@"; }

MODE=first
case "${1:-}" in
  --append) MODE=append; shift ;;
  --finish) MODE=finish; shift ;;
  -h|--help|"") sed -n 2,20p "$0"; exit 0 ;;
esac

MARKER_ARGS=()
[ -n "${MARKER_SIZE:-}" ] && MARKER_ARGS+=(--marker-size "$MARKER_SIZE")
[ -n "${CAMERA_HEIGHT:-}" ] && [ -z "${MARKER_SIZE:-}" ] && MARKER_ARGS+=(--camera-height "$CAMERA_HEIGHT")

stamp () {  # stamp <stage> <start-seconds>: record how long a stage took
  "$PY_CPU" - "$WORK/timings.json" "$1" "$2" <<'PY'
import json, os, sys, time
path, stage, start = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = json.load(open(path)) if os.path.isfile(path) else {}
d[stage] = d.get(stage, 0) + time.time() - start
json.dump(d, open(path, "w"), indent=1)
PY
}

draft_and_audit () {
  # ---- 2. draft reconstruction: feed-forward metric poses and dense points (GPU, minutes) ----------
  local t=$(date +%s)
  rm -rf "$WORK/draft"
  "$PY_DRAFT" "$MAPANYTHING/scripts/demo_colmap.py" --images_dir "$WORK/keyframes" --output_dir "$WORK/draft" --apache
  stamp draft_poses "$t"

  # ---- 3. canonical frame: gravity, Manhattan axes, metres ------------------------------------------
  t=$(date +%s)
  rm -rf "$WORK/draft_aligned"
  nt align "$WORK/draft" --out "$WORK/draft_aligned" --images "$WORK/keyframes" ${MARKER_ARGS[@]+"${MARKER_ARGS[@]}"}
  stamp align "$t"

  # ---- 4. coverage audit and recapture ticket --------------------------------------------------------
  t=$(date +%s)
  local rc=0
  nt audit "$WORK/draft_aligned" --out "$WORK/audit" --lint "$WORK/lint.json" --strict || rc=$?
  stamp audit "$t"
  # keep each pass's ticket: the before/after pair is the evidence that the loop worked
  local n=$(( $(ls -d "$WORK"/audit/history/pass_* 2>/dev/null | wc -l) + 1 ))
  mkdir -p "$WORK/audit/history/pass_$n"
  cp "$WORK/audit/recapture_ticket.png" "$WORK/audit/coverage.json" "$WORK/audit/history/pass_$n/"
  echo
  if [ "$rc" = 3 ]; then
    echo "RECAPTURE recommended. Open $WORK/audit/recapture_ticket.html, shoot the items, then:"
    echo "    bash scripts/run_pipeline.sh --append <clip.mp4> $WORK"
    echo "To train anyway:  bash scripts/run_pipeline.sh --finish $WORK"
  else
    echo "READY. Continue with:  bash scripts/run_pipeline.sh --finish $WORK"
  fi
}

if [ "$MODE" = first ] || [ "$MODE" = append ]; then
  [ $# -ge 1 ] || { echo "Give at least one video file."; exit 1; }
  VIDEOS=(); while [ $# -gt 1 ]; do VIDEOS+=("$1"); shift; done
  if [ -f "$1" ]; then VIDEOS+=("$1"); WORK=work; else WORK=$1; fi
  mkdir -p "$WORK"
  # ---- 1. capture linter and keyframes (CPU, seconds) --------------------------------------------------
  t=$(date +%s)
  LINT_ARGS=(); [ "$MODE" = append ] && LINT_ARGS+=(--append)
  rc=0; nt lint "${VIDEOS[@]}" --out "$WORK" --strict ${LINT_ARGS[@]+"${LINT_ARGS[@]}"} || rc=$?
  stamp lint "$t"
  if [ "$rc" = 2 ] && [ -z "${FORCE:-}" ]; then
    echo "The capture FAILED the linter. Read $WORK/lint_report.html and reshoot, or rerun with FORCE=1."
    exit 2
  fi
  draft_and_audit
  exit 0
fi

# ======================================= --finish ====================================================
WORK=${1:-work}
[ -d "$WORK/draft_aligned" ] || { echo "$WORK has no aligned draft yet; run the first pass before --finish."; exit 1; }

# ---- 5. final poses in the canonical frame -----------------------------------------------------------
t=$(date +%s)
rm -rf "$WORK/final"; mkdir -p "$WORK/final"
ln -sfn "$(cd "$WORK/keyframes" && pwd)" "$WORK/final/images"
if [ "$FINAL_POSES" = colmap ]; then
  rc=0
  "$PY_CPU" "$ROOT/scripts/final_poses_colmap.py" --images "$WORK/keyframes" --out "$WORK/colmap" --draft "$WORK/draft_aligned" || rc=$?
  if [ "$rc" != 0 ] && [ -z "${FORCE:-}" ]; then
    echo "COLMAP registered too few keyframes. Rerun with FINAL_POSES=draft, or FORCE=1 to accept the partial model."
    exit 4
  fi
  nt align "$WORK/colmap" --reference "$WORK/draft_aligned" --out "$WORK/final" --images "$WORK/keyframes" \
     --drop "$WORK/weak_keyframes.txt" ${MARKER_SIZE:+--marker-size "$MARKER_SIZE"}
  POSE_OPT=()
else
  # feed-forward poses, re-expressed for the full-resolution keyframes; gsplat refines them while training
  nt rescale "$WORK/draft_aligned" --images "$WORK/keyframes" --out "$WORK/final"
  cp "$WORK/draft_aligned/transform.json" "$WORK/final/transform.json"
  POSE_OPT=(--pose_opt)
fi
stamp final_poses "$t"

# ---- 6. train the splat in the metric frame (GPU, 20 to 40 minutes on a T4) ---------------------------
t=$(date +%s)
"$PY_TRAIN" "$GSPLAT/examples/simple_trainer.py" mcmc \
  --data_dir "$WORK/final" --data_factor 1 --result_dir "$WORK/train" \
  --no-normalize-world-space --test_every 8 \
  --max_steps "$STEPS" --eval_steps "$STEPS" --save_steps "$STEPS" --save_ply --ply_steps "$STEPS" \
  --strategy.cap-max "$MAX_GAUSSIANS" --disable_viewer --disable_video ${POSE_OPT[@]+"${POSE_OPT[@]}"}
stamp train "$t"
PLY=$(ls -t "$WORK"/train/ply/point_cloud_*.ply | head -1)

# ---- 7. clean, compress, nav grid, report ------------------------------------------------------------
t=$(date +%s)
mkdir -p "$WORK/package"
nt clean "$PLY" --model "$WORK/draft_aligned" --out "$WORK/package/scene_clean.ply"
"$SPLAT_TRANSFORM" -w ${SPLAT_TRANSFORM_ARGS:-} "$WORK/package/scene_clean.ply" "$WORK/package/scene.sog"
nt nav --splat "$WORK/package/scene_clean.ply" --model "$WORK/draft_aligned" --out "$WORK/package"
stamp package "$t"
nt report --work "$WORK" --scene "$SCENE_DIR" --splat "$WORK/package/scene.sog" --train "$WORK/train" --title "$TITLE"

echo
echo "Done. Preview:  cd viewer && npm install && npm run dev"
echo "Publish:        cd viewer && npm run build   (then host viewer/dist on any static host)"
