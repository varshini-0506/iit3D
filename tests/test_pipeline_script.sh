#!/usr/bin/env bash
# Runs scripts/run_pipeline.sh end to end on the synthetic room with the two GPU stages replaced by
# stand-ins (tests/stubs). Everything else is real: the linter, alignment, audit, the recapture
# loop, COLMAP bundle adjustment with draft-guided matching, clean-up, SOG compression, the nav
# grid and the report. Needs: pip install -r requirements.txt pycolmap, and
# npm install --prefix third_party @playcanvas/splat-transform@3. Takes about four minutes.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${1:-$ROOT/work_test}
export PYTHONPATH="$ROOT"
export PY_CPU=${PY_CPU:-python3} PY_DRAFT=${PY_DRAFT:-python3} PY_TRAIN=${PY_TRAIN:-python3}
export MAPANYTHING="$ROOT/tests/stubs/map-anything" GSPLAT="$ROOT/tests/stubs/gsplat"
export MARKER_SIZE=0.16 TITLE="Pipeline test (synthetic)" SPLAT_TRANSFORM_ARGS="-g cpu" STEPS=30000
export SCENE_DIR="$WORK/scene"
rm -rf "$WORK"; mkdir -p "$WORK/clips"

python3 - "$WORK/clips" <<'PY'
import sys
from nammatwin import synth
room = synth.Room()
t, poses = synth.walk_trajectory(room, 10.0)
synth.write_video(room, f"{sys.argv[1]}/walk.mp4", t, poses, blur=(60.0, 61.5))
t, poses = synth.recapture_trajectory(room, 10.0)
synth.write_video(room, f"{sys.argv[1]}/topup.mp4", t, poses, seed=2)
PY

bash "$ROOT/scripts/run_pipeline.sh" "$WORK/clips/walk.mp4" "$WORK" | tee "$WORK/pass1.log"
grep -q "RECAPTURE recommended" "$WORK/pass1.log"
bash "$ROOT/scripts/run_pipeline.sh" --append "$WORK/clips/topup.mp4" "$WORK" | tee "$WORK/pass2.log"
grep -q "^READY" "$WORK/pass2.log"
bash "$ROOT/scripts/run_pipeline.sh" --finish "$WORK" | tee "$WORK/finish.log"

python3 - "$WORK" "$SCENE_DIR" <<'PY'
import json, os, sys
work, scene = sys.argv[1:3]
rep = json.load(open(os.path.join(scene, "report.json")))
man = json.load(open(os.path.join(scene, "manifest.json")))
assert man["splat"] == "scene.sog" and os.path.getsize(os.path.join(scene, "scene.sog")) > 10_000
g = rep["geometry"]
assert abs(g["model_scale_error_pct"] - 3.0) < 0.5, g          # the stub draft was 3% out; the marker caught it
assert g["draft_vs_final_pose_rmse_cm"] < 3.0, g                 # COLMAP and the draft agree
assert abs(rep["space"]["floor_area_m2"] - 19.0) < 1.5, rep["space"]   # true free floor of the room is 19.0 m2
assert rep["coverage"]["verdict"] == "READY"
print("pipeline script test passed:", json.dumps({k: rep[k] for k in ("geometry", "space", "asset")}))
PY
