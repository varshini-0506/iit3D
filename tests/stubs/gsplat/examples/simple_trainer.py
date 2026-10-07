#!/usr/bin/env python3
"""Stand-in for gsplat's examples/simple_trainer.py, for testing the pipeline script without a GPU.

Accepts the flags the pipeline passes, checks that the dataset is laid out the way gsplat's COLMAP
parser expects, and writes a splat of the synthetic room in the frame of the poses it was given,
which is what real training would produce. The fidelity numbers it writes are placeholders.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from _truth import keyframe_poses  # noqa: E402

from nammatwin import colmap_io as cio, synth  # noqa: E402
from nammatwin.geometry import umeyama  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("strategy", choices=["default", "mcmc"])
ap.add_argument("--data_dir", required=True)
ap.add_argument("--data_factor", type=int, default=4)
ap.add_argument("--result_dir", required=True)
ap.add_argument("--no-normalize-world-space", action="store_true")
ap.add_argument("--test_every", type=int, default=8)
ap.add_argument("--max_steps", type=int, default=30000)
ap.add_argument("--eval_steps", type=int, nargs="+")
ap.add_argument("--save_steps", type=int, nargs="+")
ap.add_argument("--save_ply", action="store_true")
ap.add_argument("--ply_steps", type=int, nargs="+")
ap.add_argument("--strategy.cap-max", dest="cap_max", type=int)
ap.add_argument("--disable_viewer", action="store_true")
ap.add_argument("--disable_video", action="store_true")
ap.add_argument("--pose_opt", action="store_true")
a = ap.parse_args()

assert a.no_normalize_world_space, "the splat must be trained in the frame it was given"
assert a.data_factor == 1 and a.save_ply
sparse = os.path.join(a.data_dir, "sparse", "0")
assert os.path.isfile(os.path.join(sparse, "images.bin")), "gsplat reads data_dir/sparse/0/*.bin"
model = cio.read_model(sparse)
for im in model.images.values():
    assert os.path.isfile(os.path.join(a.data_dir, "images", im.name)), f"missing image {im.name}"

# where the given poses put the room: fit ground-truth camera centres to the dataset's
room, names, c2ws = keyframe_poses(os.path.dirname(os.path.abspath(a.data_dir)))
truth = dict(zip(names, [m[:3, 3] for m in c2ws]))
src = np.array([truth[im.name] for im in model.sorted_images()])
s, R, t = umeyama(src, model.centers())
step = a.max_steps - 1
os.makedirs(os.path.join(a.result_dir, "ply"), exist_ok=True)
os.makedirs(os.path.join(a.result_dir, "stats"), exist_ok=True)
n = synth.write_splat_ply(room, os.path.join(a.result_dir, "ply", f"point_cloud_{step}.ply"), transform=(s, R, t))
with open(os.path.join(a.result_dir, "stats", f"val_step{step:04d}.json"), "w") as f:
    json.dump({"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "ellipse_time": 0.0, "num_GS": n}, f)
with open(os.path.join(a.result_dir, "stats", f"train_step{step:04d}_rank0.json"), "w") as f:
    json.dump({"mem": 0.0, "ellipse_time": 0.0, "num_GS": n}, f)
print(f"[stub train] {len(model.images)} images, wrote {n} Gaussians")
