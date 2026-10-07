"""Twin report card: every stage's numbers in one file, published next to the splat.

The web viewer shows this card beside the scene and adds the two numbers that
can only be measured on the viewer's own device: load time and frame rate.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time


def _load(path: str) -> dict | None:
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def _gaussian_count(path: str) -> int | None:
    if not path.lower().endswith(".ply"):
        return None
    with open(path, "rb") as f:
        for _ in range(60):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def _train_stats(train_dir: str | None) -> dict | None:
    if not train_dir:
        return None
    vals = sorted(glob.glob(os.path.join(train_dir, "stats", "val_step*.json")))
    if not vals:
        return None
    v = _load(vals[-1]) or {}
    out = dict(step=int(os.path.basename(vals[-1])[8:-5]) + 1)
    for k in ("psnr", "ssim", "lpips"):
        if k in v:
            out[k] = round(float(v[k]), 4)
    if "num_GS" in v:
        out["gaussians"] = int(v["num_GS"])
    trains = sorted(glob.glob(os.path.join(train_dir, "stats", "train_step*.json")))
    if trains:
        t = _load(trains[-1]) or {}
        if "ellipse_time" in t:
            out["train_minutes"] = round(float(t["ellipse_time"]) / 60.0, 1)
    return out


def build(work: str, splat: str | None = None, train_dir: str | None = None, title: str = "NammaTwin scene") -> dict:
    lint = _load(os.path.join(work, "lint.json"))
    cov = _load(os.path.join(work, "audit", "coverage.json"))
    draft_tf = _load(os.path.join(work, "draft_aligned", "transform.json"))
    final_tf = _load(os.path.join(work, "final", "transform.json"))
    nav = _load(os.path.join(work, "package", "nav.json")) or _load(os.path.join(work, "audit", "nav.json"))
    timings = _load(os.path.join(work, "timings.json"))
    train = _train_stats(train_dir or os.path.join(work, "train"))

    rep: dict = dict(title=title, generated=time.strftime("%Y-%m-%d %H:%M"), pipeline="NammaTwin 0.1")
    if lint:
        rep["capture"] = dict(
            verdict=lint["verdict"], score=lint["score"], clips=len(lint["clips"]),
            duration_s=lint["total_duration_s"], keyframes=lint["total_keyframes"],
            blurry_pct=round(100 * sum(c["fractions"]["blurry"] * c["duration_s"] for c in lint["clips"])
                             / max(lint["total_duration_s"], 1e-6), 1),
            too_fast_pct=round(100 * sum(c["fractions"]["fast"] * c["duration_s"] for c in lint["clips"])
                               / max(lint["total_duration_s"], 1e-6), 1),
            loop_closed=lint["clips"][0]["loop_closure"]["closed"])
    if cov:
        s = cov["summary"]
        rep["coverage"] = dict(verdict=s["verdict"], walk_view_quality_pct=s["walk_view_quality_pct"],
                               ceiling_quality_pct=s["by_band"]["high_pct"],
                               boundary_closure_pct=s["boundary_closure_pct"], open_items=s["items"],
                               median_views=s["median_views"], median_baseline_deg=s["median_baseline_deg"])
    tf = draft_tf or final_tf
    if tf:
        d = tf["diagnostics"]
        geo = dict(scale_source=tf["scale_source"], floor_rms_cm=round(100 * d.get("floor_rms_m", 0), 2),
                   camera_height_m=d.get("camera_height_m"))
        if "model_scale_error_pct" in d:
            geo["model_scale_error_pct"] = d["model_scale_error_pct"]
        mk = d.get("marker_check")
        if mk:
            geo.update(marker_error_pct=mk["scale_error_pct"], marker_error_cm_per_m=mk["error_cm_per_m"],
                       marker_sightings=mk["observations"], marker_independent=mk.get("independent", False))
        if final_tf and final_tf["scale_source"] == "reference_model":
            geo["draft_vs_final_pose_rmse_cm"] = round(100 * final_tf["diagnostics"]["trajectory_rmse_m"], 2)
        rep["geometry"] = geo
    if nav:
        rep["space"] = dict(nav["stats"])
        if "closure" in nav:
            rep["space"]["boundary_closure_pct"] = nav["closure"]["closure_pct"]
    if train:
        rep["fidelity"] = train
    if splat and os.path.isfile(splat):
        asset = dict(file=os.path.basename(splat), size_mb=round(os.path.getsize(splat) / 1e6, 2))
        n = _gaussian_count(splat)
        for sibling in ("scene_clean.ply", "scene.ply"):  # compressed formats: count from the PLY they came from
            cand = os.path.join(os.path.dirname(splat), sibling)
            if n is None and os.path.isfile(cand):
                n = _gaussian_count(cand)
        n = n or (train or {}).get("gaussians")
        if n:
            asset["gaussians"] = int(n)
        rep["asset"] = asset
    if timings:
        rep["timings_min"] = {k: round(v / 60.0, 1) for k, v in timings.items()}
    return rep


def run(work: str, scene: str, splat: str | None = None, train_dir: str | None = None,
        title: str = "NammaTwin scene") -> dict:
    """Write report.json and publish everything the viewer needs into `scene`."""
    os.makedirs(scene, exist_ok=True)
    rep = build(work, splat, train_dir, title)
    nav_dir = os.path.join(work, "package") if os.path.isfile(os.path.join(work, "package", "nav.json")) \
        else os.path.join(work, "audit")
    for name in ("nav.json", "nav.png", "floorplan.png"):
        src = os.path.join(nav_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(scene, name))
    for name in ("coverage.png", "recapture_ticket.png"):
        src = os.path.join(work, "audit", name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(scene, name))
    manifest = dict(title=title, up="z", units="m", nav="nav.json", report="report.json")
    if splat and os.path.isfile(splat):
        dst = os.path.join(scene, os.path.basename(splat))
        if os.path.abspath(dst) != os.path.abspath(splat):
            shutil.copy2(splat, dst)
        manifest["splat"] = os.path.basename(splat)
    if os.path.isfile(os.path.join(scene, "coverage.png")):
        manifest["coverage"] = "coverage.png"
        # the coverage overlay is drawn on the audit's grid, which can differ from the final grid
        a = _load(os.path.join(work, "audit", "nav.json"))
        if a:
            manifest["coverage_grid"] = {k: a[k] for k in ("cell", "x0", "y0", "width", "height")}
    with open(os.path.join(scene, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(scene, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[report] wrote {os.path.join(scene, 'report.json')} and manifest.json")
    for k, v in rep.items():
        if isinstance(v, dict):
            print(f"[report]   {k}: {v}")
    return rep
