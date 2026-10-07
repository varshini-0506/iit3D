#!/usr/bin/env python3
"""Final camera poses: COLMAP bundle adjustment on the full-resolution keyframes.

    python scripts/final_poses_colmap.py --images work/keyframes --out work/colmap \
        --draft work/draft_aligned

With --draft, image pairs come from the draft poses (see nammatwin/pairs.py),
which is much faster than exhaustive matching and links loop closures and
recapture clips. The output is an ordinary COLMAP model in arbitrary units;
`python -m nammatwin align --reference work/draft_aligned` then moves it into
the metric, gravity-aligned frame.

Needs: pip install pycolmap  (CPU wheel is enough; a CUDA build is faster).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory of full-resolution keyframes")
    ap.add_argument("--out", required=True, help="output directory (database.db and sparse/0 are written here)")
    ap.add_argument("--draft", help="aligned draft model, used to choose which image pairs to match")
    ap.add_argument("--matcher", choices=["auto", "guided", "sequential", "exhaustive"], default="auto")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--neighbours", type=int, default=25, help="pairs per image for guided matching")
    ap.add_argument("--overlap", type=int, default=20, help="frames of overlap for sequential matching")
    a = ap.parse_args()

    import pycolmap

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    db = os.path.join(a.out, "database.db")
    if os.path.exists(db):
        os.remove(db)
    names = sorted(n for n in os.listdir(a.images) if n.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"[colmap] {len(names)} images in {a.images}")

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = a.camera_model
    pycolmap.extract_features(db, a.images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    print(f"[colmap] features extracted in {time.time() - t0:.0f} s")

    matcher = a.matcher
    if matcher == "auto":
        matcher = "guided" if a.draft and hasattr(pycolmap, "match_image_pairs") else \
            ("exhaustive" if len(names) <= 150 else "sequential")
    t1 = time.time()
    if matcher == "guided":
        from nammatwin import colmap_io as cio, pairs as pr
        draft = cio.read_model(a.draft)
        have = set(names)
        pairs = [(x, y) for x, y in pr.pose_guided_pairs(draft, k=a.neighbours) if x in have and y in have]
        pair_file = os.path.join(a.out, "pairs.txt")
        pr.write_pairs(pairs, pair_file)
        opts = pycolmap.ImportedPairingOptions()
        opts.match_list_path = pair_file
        pycolmap.match_image_pairs(db, pairing_options=opts)
        full = len(names) * (len(names) - 1) // 2
        print(f"[colmap] matched {len(pairs)} draft-guided pairs ({100 * len(pairs) / max(full, 1):.0f}% of exhaustive)")
    elif matcher == "exhaustive":
        pycolmap.match_exhaustive(db)
    else:
        opts = pycolmap.SequentialPairingOptions()
        opts.overlap = a.overlap
        opts.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=opts)
    print(f"[colmap] matching ({matcher}) took {time.time() - t1:.0f} s")

    raw = os.path.join(a.out, "sparse_raw")
    shutil.rmtree(raw, ignore_errors=True)
    os.makedirs(raw)
    t2 = time.time()
    maps = pycolmap.incremental_mapping(db, a.images, raw)
    if not maps:
        print("[colmap] FAILED: no model could be built. Use the draft poses instead (see README, 'If COLMAP fails').")
        return 2
    best = max(maps.values(), key=lambda r: r.num_reg_images())
    final = os.path.join(a.out, "sparse", "0")
    shutil.rmtree(final, ignore_errors=True)
    os.makedirs(final)
    best.write(final)
    reg = best.num_reg_images()
    summary = dict(images=len(names), registered=reg, registered_pct=round(100 * reg / max(len(names), 1), 1),
                   points=best.num_points3D(), mean_reprojection_error_px=round(best.compute_mean_reprojection_error(), 3),
                   matcher=matcher, models_found=len(maps), seconds=round(time.time() - t0, 1))
    with open(os.path.join(a.out, "colmap_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[colmap] mapping took {time.time() - t2:.0f} s; {summary}")
    if reg < 0.9 * len(names):
        print(f"[colmap] WARNING: only {reg} of {len(names)} keyframes registered. Blank walls and fast pans break "
              "feature matching. Consider the draft poses instead (see README, 'If COLMAP fails').")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
