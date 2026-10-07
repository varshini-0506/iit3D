"""Command line: python -m nammatwin <stage> ..."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _add_lint(sub):
    p = sub.add_parser("lint", help="grade a walkthrough video and extract keyframes")
    p.add_argument("videos", nargs="+")
    p.add_argument("--out", required=True, help="work directory (keyframes/ and reports are written here)")
    p.add_argument("--append", action="store_true", help="add recapture clips to an existing work directory")
    p.add_argument("--step", type=float, default=0.07, help="parallax between keyframes, fraction of image width")
    p.add_argument("--max-keyframes", type=int, default=400)
    p.add_argument("--max-side", type=int, default=1600, help="longest side of saved keyframes, pixels")
    p.add_argument("--strict", action="store_true", help="exit with code 2 on a FAIL verdict")


def _add_align(sub):
    p = sub.add_parser("align", help="put a COLMAP-format model into the metric, gravity-aligned frame")
    p.add_argument("model", help="directory holding the model (or its parent with sparse/)")
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, help="metres per model unit, if you know it")
    p.add_argument("--camera-height", type=float, help="set the scale so the median camera is this high (metres)")
    p.add_argument("--reference", help="aligned model to copy the frame from (matched by image name)")
    p.add_argument("--marker-size", type=float, help="printed edge length of the ArUco marker, metres")
    p.add_argument("--marker-mode", choices=["auto", "set", "check"], default="auto",
                   help="auto: use the marker for scale when it was seen well; set: always; check: only report")
    p.add_argument("--drop", help="text file of image names to leave out (e.g. work/weak_keyframes.txt)")
    p.add_argument("--marker-id", type=int)
    p.add_argument("--marker-dict", default="DICT_4X4_50")
    p.add_argument("--images", help="keyframe directory for marker detection (default: images/ beside the model)")


def _add_audit(sub):
    p = sub.add_parser("audit", help="coverage heatmap and recapture ticket from an aligned draft")
    p.add_argument("model", help="aligned model directory")
    p.add_argument("--out", required=True)
    p.add_argument("--lint", help="lint.json, to quote clip times on the ticket")
    p.add_argument("--min-views", type=int, default=4)
    p.add_argument("--min-baseline", type=float, default=12.0, help="degrees")
    p.add_argument("--max-range", type=float, default=6.0, help="metres")
    p.add_argument("--strict", action="store_true", help="exit with code 3 when recapture is recommended")


def _add_nav(sub):
    p = sub.add_parser("nav", help="walkable grid and floor plan from the final splat")
    p.add_argument("--splat", help="trained 3DGS .ply in the aligned frame")
    p.add_argument("--model", required=True, help="aligned model (camera path, and points if no splat is given)")
    p.add_argument("--out", required=True)
    p.add_argument("--cell", type=float, default=0.05)
    p.add_argument("--min-opacity", type=float, default=0.5)


def _add_clean(sub):
    p = sub.add_parser("clean", help="crop a trained splat to the room and drop debris")
    p.add_argument("splat", help="trained 3DGS .ply in the aligned frame")
    p.add_argument("--model", required=True, help="aligned model (defines the room box)")
    p.add_argument("--out", required=True, help="output .ply")
    p.add_argument("--margin", type=float, default=0.6, help="metres kept beyond the scanned extent")
    p.add_argument("--min-opacity", type=float, default=0.02)


def _add_rescale(sub):
    p = sub.add_parser("rescale", help="re-express a model's intrinsics for full-resolution keyframes")
    p.add_argument("model")
    p.add_argument("--images", required=True, help="directory of the full-resolution keyframes")
    p.add_argument("--out", required=True)


def _add_report(sub):
    p = sub.add_parser("report", help="gather every stage's numbers into the viewer's report card")
    p.add_argument("--work", required=True, help="work directory")
    p.add_argument("--scene", required=True, help="viewer scene directory to fill (e.g. viewer/public/scene)")
    p.add_argument("--splat", help="packaged splat file to publish (.sog / .spz / .ply)")
    p.add_argument("--train", help="gsplat result directory (for PSNR / SSIM / LPIPS)")
    p.add_argument("--title", default="NammaTwin scene")


def _add_marker(sub):
    p = sub.add_parser("marker", help="write a printable ArUco marker PNG")
    p.add_argument("--out", default="marker.png")
    p.add_argument("--id", type=int, default=0)
    p.add_argument("--dict", default="DICT_4X4_50")


def _add_sample(sub):
    p = sub.add_parser("sample", help="generate the synthetic sample scene and run every CPU stage on it")
    p.add_argument("--out", required=True, help="work directory for the sample run")
    p.add_argument("--scene", help="also publish to this viewer scene directory")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nammatwin", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for add in (_add_lint, _add_align, _add_audit, _add_nav, _add_clean, _add_rescale, _add_report, _add_marker,
                _add_sample):
        add(sub)
    a = ap.parse_args(argv)

    if a.cmd == "lint":
        from . import lint
        cfg = lint.LintConfig(keyframe_step=a.step, max_keyframes=a.max_keyframes, max_side=a.max_side)
        res = lint.run(a.videos, a.out, cfg, append=a.append)
        return 2 if a.strict and res["verdict"] == "FAIL" else 0

    if a.cmd == "align":
        from . import align
        align.run(a.model, a.out, scale=a.scale, camera_height=a.camera_height, reference=a.reference,
                  marker_size=a.marker_size, marker_mode=a.marker_mode, marker_id=a.marker_id,
                  marker_dict=a.marker_dict, images_dir=a.images, drop=a.drop)
        return 0

    if a.cmd == "audit":
        from . import coverage
        cfg = coverage.AuditConfig(min_views=a.min_views, min_baseline_deg=a.min_baseline, max_range=a.max_range)
        res = coverage.run(a.model, a.out, lint_json=a.lint, cfg=cfg)
        return 3 if a.strict and res["summary"]["verdict"] != "READY" else 0

    if a.cmd == "nav":
        import numpy as np
        from . import colmap_io as cio, navgrid
        model = cio.read_model(a.model)
        ims = model.sorted_images()
        centers = np.array([im.center() for im in ims])
        if a.splat:
            xyz, _ = cio.read_splat_points(a.splat, a.min_opacity)
            print(f"[nav] {len(xyz)} Gaussians above opacity {a.min_opacity} from {a.splat}")
        else:
            xyz = model.xyz
        f = ims[0].R()[2, :2]
        grid = navgrid.build(xyz, centers, f / max(np.linalg.norm(f), 1e-9), navgrid.NavConfig(cell=a.cell))
        fronts, closure = navgrid.frontiers(grid)
        navgrid.save(grid, a.out, extra=dict(closure=closure))
        print(f"[nav] {grid.stats}; boundary closure {closure['closure_pct']}%")
        return 0

    if a.cmd == "clean":
        import numpy as np
        from . import colmap_io as cio, splat
        model = cio.read_model(a.model)
        lo, hi = splat.room_bounds(model.xyz, model.centers(), a.margin)
        stats = splat.clean(a.splat, a.out, lo, hi, a.min_opacity)
        print(f"[clean] room box {np.round(lo, 2).tolist()} to {np.round(hi, 2).tolist()} m")
        print(f"[clean] {stats}")
        return 0

    if a.cmd == "rescale":
        import cv2
        from . import align, colmap_io as cio
        model = cio.read_model(a.model)
        name = model.sorted_images()[0].name
        img = cv2.imread(os.path.join(a.images, name))
        if img is None:
            print(f"Cannot read {os.path.join(a.images, name)}", file=sys.stderr)
            return 1
        h, w = img.shape[:2]
        out = align.rescale_intrinsics(model, w, h)
        cio.write_model(out, os.path.join(a.out, "sparse", "0"))
        print(f"[rescale] intrinsics now match {w}x{h} images; wrote {os.path.join(a.out, 'sparse', '0')}")
        return 0

    if a.cmd == "report":
        from . import report
        report.run(a.work, a.scene, splat=a.splat, train_dir=a.train, title=a.title)
        return 0

    if a.cmd == "marker":
        from . import scale_marker
        scale_marker.make_marker_png(a.out, a.id, a.dict)
        print(f"Wrote {a.out}. Print it, measure the black square's edge with a ruler, and pass that as --marker-size.")
        return 0

    if a.cmd == "sample":
        from . import sample
        sample.run(a.out, a.scene)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
