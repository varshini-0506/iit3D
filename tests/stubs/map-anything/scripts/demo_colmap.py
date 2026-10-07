#!/usr/bin/env python3
"""Stand-in for MapAnything's scripts/demo_colmap.py, for testing the pipeline script without a GPU.

Same command line and same output layout as the real script (images/ at the model's inference
resolution, sparse/ in COLMAP format), but the poses and points come from the synthetic room's
ground truth, handed over in a tilted frame that is 3% out of scale.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from _truth import SIZE, keyframe_poses  # noqa: E402

from nammatwin import align, colmap_io as cio, synth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images_dir", required=True)
ap.add_argument("--output_dir", required=True)
ap.add_argument("--apache", action="store_true")
a = ap.parse_args()

room, names, c2ws = keyframe_poses(os.path.dirname(os.path.abspath(a.images_dir)))
K = synth.intrinsics(*SIZE)
model = synth.draft_model(room, K, SIZE, c2ws, names)

# MapAnything resizes each image to cover a fixed 518-wide target and centre-crops it
tw, th = 518, 294
k = max(tw / SIZE[0], th / SIZE[1])
rw, rh = int(round(SIZE[0] * k)), int(round(SIZE[1] * k))
ox, oy = (rw - tw) // 2, (rh - th) // 2
os.makedirs(os.path.join(a.output_dir, "images"), exist_ok=True)
for n in names:
    img = cv2.resize(cv2.imread(os.path.join(a.images_dir, n)), (rw, rh), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(a.output_dir, "images", n), img[oy:oy + th, ox:ox + tw])
cam = model.cameras[1]
fx, fy, cx, cy = cam.fxfycxcy()
model.cameras[1] = cio.Camera(1, "PINHOLE", tw, th, np.array([fx * k, fy * k, cx * k - ox, cy * k - oy]))

draft = align.apply_similarity(model, *synth.random_similarity(scale=1.03))
cio.write_model(draft, os.path.join(a.output_dir, "sparse"))
print(f"[stub draft] {len(names)} cameras, {len(draft.xyz)} points -> {a.output_dir}")
