"""End-to-end run of every CPU stage on the synthetic room.

This is the repository's self-test and its sample data. It walks the same loop
a real capture does:

    1. render a two-pass walkthrough (with a blurred stretch and a whip-pan)
    2. lint it and pick keyframes
    3. build a draft model: exact poses, but only surface those keyframes saw,
       handed over tilted, rotated, shifted and 3% out of scale
    4. align it (gravity, Manhattan, origin) and correct the scale from the marker
    5. audit coverage -> recapture ticket (the alcove was never entered)
    6. render the top-up clip the ticket asks for, lint --append, redo 3 to 5
    7. write a splat of the room in the aligned frame, build the nav grid,
       publish scene + report card for the web viewer

The GPU stages (MapAnything, gsplat) are replaced here by ground truth; the
code around them is exactly what runs on a real capture.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import numpy as np

from . import align, colmap_io as cio, coverage, lint, navgrid, report, synth

SIZE = (640, 360)
FPS = 10.0
MARKER = 0.16


def _link_dir(src: str, dst: str) -> None:
    """Point dst at src. A directory symlink when the OS allows it; otherwise one hard link per file.

    Windows refuses os.symlink unless Developer Mode or an admin shell is on (WinError 1314).
    A directory junction is not a safe substitute: shutil.rmtree follows it and would delete the
    original keyframes. Per-file hard links stay on the same volume and unlink cleanly.
    """
    src = os.path.abspath(src)
    try:
        os.symlink(src, dst, target_is_directory=True)
        return
    except OSError:
        pass
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.isdir(s):
            _link_dir(s, d)
            continue
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)


def _draft(room, K, lint_res, pose_tables, sim, out_dir, kf_dir):
    """Stand-in for the feed-forward draft: GT poses of the keyframes, seen-only points, arbitrary frame."""
    c2ws, names = [], []
    for clip in lint_res["clips"]:
        for kf in clip["keyframe_list"]:
            c2ws.append(pose_tables[clip["clip"]][kf["frame"]])
            names.append(kf["name"])
    order = np.argsort(names)
    gt = synth.draft_model(room, K, SIZE, [c2ws[i] for i in order], [names[i] for i in order])
    draft = align.apply_similarity(gt, *sim)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    cio.write_model(draft, os.path.join(out_dir, "sparse"))
    _link_dir(kf_dir, os.path.join(out_dir, "images"))


def _compress(ply: str) -> str:
    """Compress to SOG when splat-transform is installed; otherwise publish the PLY as it is."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tool = shutil.which("splat-transform") or os.path.join(here, "third_party", "node_modules", ".bin", "splat-transform")
    if not os.path.isfile(tool):
        print("[sample] splat-transform not found; publishing the uncompressed PLY")
        return ply
    sog = os.path.splitext(ply)[0] + ".sog"
    try:
        subprocess.run([tool, "-q", "-w", "-g", "cpu", ply, sog], check=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as e:
        print(f"[sample] splat-transform failed ({e}); publishing the uncompressed PLY")
        return ply
    print(f"[sample] compressed {os.path.getsize(ply) / 1e6:.1f} MB PLY to {os.path.getsize(sog) / 1e6:.2f} MB SOG")
    return sog


def run(work: str, scene: str | None = None) -> dict:
    t_start = time.time()
    os.makedirs(work, exist_ok=True)
    room = synth.Room()
    timings = {}

    # 1. the walkthrough -------------------------------------------------------
    t0 = time.time()
    t, poses = synth.walk_trajectory(room, FPS)
    video = os.path.join(work, "walk.mp4")
    K = synth.write_video(room, video, t, poses, SIZE, FPS, blur=(60.0, 61.5))
    print(f"[sample] rendered {len(t)} frames ({t[-1]:.0f} s) to {video}")

    # 2. lint ------------------------------------------------------------------
    cfg = lint.LintConfig(max_side=max(SIZE))
    res = lint.run([video], work, cfg)
    timings["lint"] = time.time() - t0
    pose_tables = {0: poses}
    sim = synth.random_similarity(scale=1.03)  # a "metric" draft that is 3% out
    kf_dir = os.path.join(work, "keyframes")

    def reconstruct_and_audit(tag: str) -> dict:
        t1 = time.time()
        _draft(room, K, res, pose_tables, sim, os.path.join(work, "draft"), kf_dir)
        aligned_dir = os.path.join(work, "draft_aligned")
        if os.path.isdir(aligned_dir):
            shutil.rmtree(aligned_dir)
        align.run(os.path.join(work, "draft"), aligned_dir, marker_size=MARKER)
        timings["align"] = time.time() - t1
        t1 = time.time()
        audit = coverage.run(aligned_dir, os.path.join(work, "audit"), lint_json=os.path.join(work, "lint.json"))
        timings["audit"] = time.time() - t1
        for name in ("recapture_ticket.png", "recapture_ticket.html", "coverage.json"):
            shutil.copy2(os.path.join(work, "audit", name), os.path.join(work, "audit", f"{tag}_{name}"))
        return audit

    # 3-5. first draft, alignment, audit -----------------------------------------
    audit1 = reconstruct_and_audit("pass1")

    # 6. the recapture loop ------------------------------------------------------
    t2, poses2 = synth.recapture_trajectory(room, FPS)
    video2 = os.path.join(work, "recapture_01.mp4")
    synth.write_video(room, video2, t2, poses2, SIZE, FPS, seed=2)
    res = lint.run([video2], work, cfg, append=True)
    pose_tables[1] = poses2
    audit2 = reconstruct_and_audit("pass2")

    # 7. splat, nav grid, report ---------------------------------------------------
    # (on a real capture the splat is trained by gsplat from work/final; here it is written directly)
    with open(os.path.join(work, "draft_aligned", "transform.json")) as f:
        tf = json.load(f)
    s, R, tr = align.compose(tf["scale"], np.array(tf["rotation"]), np.array(tf["translation"]), *sim)
    pkg = os.path.join(work, "package")
    os.makedirs(pkg, exist_ok=True)
    splat = os.path.join(pkg, "scene.ply")
    n = synth.write_splat_ply(room, splat, spacing=0.03, transform=(s, R, tr))
    print(f"[sample] wrote synthetic splat with {n} Gaussians")
    aligned = cio.read_model(os.path.join(work, "draft_aligned"))
    ims = aligned.sorted_images()
    centers = np.array([im.center() for im in ims])
    xyz, _ = cio.read_splat_points(splat)
    f0 = ims[0].R()[2, :2]
    grid = navgrid.build(xyz, centers, f0 / np.linalg.norm(f0))
    _, closure = navgrid.frontiers(grid)
    navgrid.save(grid, pkg, extra=dict(closure=closure))
    timings["total"] = time.time() - t_start
    with open(os.path.join(work, "timings.json"), "w") as f:
        json.dump(timings, f, indent=1)
    rep = None
    if scene:
        rep = report.run(work, scene, splat=_compress(splat), title="Sample room (synthetic)")

    s1, s2 = audit1["summary"], audit2["summary"]
    print("\n[sample] closed loop on the synthetic room")
    print(f"[sample]   before recapture: walk-view surface {s1['walk_view_quality_pct']}%, "
          f"closure {s1['boundary_closure_pct']}%, {s1['items']} ticket item(s) -> {s1['verdict']}")
    print(f"[sample]   after recapture:  walk-view surface {s2['walk_view_quality_pct']}%, "
          f"closure {s2['boundary_closure_pct']}%, {s2['items']} ticket item(s) -> {s2['verdict']}")
    return dict(lint=res, audit_before=audit1, audit_after=audit2, transform=tf, nav=grid.stats, report=rep,
                room=dict(lx=room.lx, ly=room.ly, h=room.h, alcove=room.alcove, marker=MARKER))
