"""Tests for the CPU-side pipeline. Run with:  python -m pytest tests -q

Ground truth comes from the synthetic room in nammatwin/synth.py. The session
fixture runs the whole sample loop once (about 40 seconds); the rest are quick.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nammatwin import align, colmap_io as cio, coverage, navgrid, pairs, sample, splat, synth  # noqa: E402
from nammatwin.geometry import rot_z, umeyama  # noqa: E402


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="session")
def run(tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    out = sample.run(str(work), str(work / "scene"))
    out["work"] = str(work)
    return out


def _tiny_model(seed=0):
    """A small room model with ground-truth cameras, in the true (Z-up, metric) frame."""
    room = synth.Room()
    K = synth.intrinsics(320, 180)
    _, poses = synth.walk_trajectory(room, fps=2.0, fast_pan=None)
    names = [f"f{i:04d}.jpg" for i in range(len(poses))]
    return room, synth.draft_model(room, K, (320, 180), poses, names, spacing=0.08, seed=seed)


# ------------------------------------------------------------------ io and geometry

def test_colmap_roundtrip_bin_and_txt(tmp_path):
    _, m = _tiny_model()
    for ext in (".bin", ".txt"):
        d = str(tmp_path / ext.strip("."))
        cio.write_model(m, d, ext)
        r = cio.read_model(d)
        assert len(r.images) == len(m.images) and len(r.xyz) == len(m.xyz)
        a, b = m.sorted_images()[5], r.sorted_images()[5]
        assert a.name == b.name
        assert np.allclose(a.qvec, b.qvec, atol=1e-9) and np.allclose(a.tvec, b.tvec, atol=1e-9)
        assert np.allclose(r.xyz, m.xyz, atol=1e-6) and (r.rgb == m.rgb).all()
        assert np.allclose(r.cameras[1].params, m.cameras[1].params)


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3))
    s, R, t = 2.5, rot_z(0.7), np.array([1.0, -2.0, 0.5])
    s2, R2, t2 = umeyama(src, s * src @ R.T + t)
    assert abs(s2 - s) < 1e-9 and np.allclose(R2, R, atol=1e-9) and np.allclose(t2, t, atol=1e-9)


def test_similarity_keeps_every_reprojection():
    _, m = _tiny_model()
    s, R, t = synth.random_similarity(seed=4)
    m2 = align.apply_similarity(m, s, R, t)
    K = m.cameras[1].K()
    for a, b in list(zip(m.sorted_images(), m2.sorted_images()))[::7]:
        pa = (m.xyz[:200] @ a.R().T + a.tvec) @ K.T
        pb = (m2.xyz[:200] @ b.R().T + b.tvec) @ K.T
        front = pa[:, 2] > 0.1
        assert np.allclose(pa[front, :2] / pa[front, 2:], pb[front, :2] / pb[front, 2:], atol=1e-6)


def test_intrinsics_rescale_inverts_resize_and_centre_crop():
    cam = cio.Camera(1, "PINHOLE", 518, 294, np.array([380.0, 380.0, 259.0, 147.0]))
    m = cio.Model({1: cam}, {}, np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0, np.int64), np.zeros(0))
    big = align.rescale_intrinsics(m, 1600, 900).cameras[1]
    k = max(518 / 1600, 294 / 900)
    fx, fy, cx, cy = big.fxfycxcy()
    assert abs(fx - 380 / k) < 1e-9 and abs(cx - 800) < 1e-6 and abs(cy - 450) < 1e-6
    # a ray through a full-size pixel must land on the matching pixel of the cropped small image
    u, v = 1234.0, 321.0
    x, y = (u - cx) / fx, (v - cy) / fy
    assert abs(380 * x + 259 - (u * k - (1600 * k - 518) / 2)) < 1e-6
    assert abs(380 * y + 147 - (v * k - (900 * k - 294) / 2)) < 1e-6


# ------------------------------------------------------------------ alignment

@pytest.mark.parametrize("seed", [11, 12, 13])
def test_alignment_finds_gravity_walls_and_scale(seed):
    room, gt = _tiny_model(seed)
    s, R, t = synth.random_similarity(seed=seed)  # tilted, spun, shifted, scaled by 0.2 to 3
    messy = align.apply_similarity(gt, s, R, t)
    tf = align.canonicalise(messy, camera_height=1.45)
    out = align.apply_similarity(messy, tf["s"], tf["R"], tf["t"])
    z = out.xyz[:, 2]
    floor = z[np.abs(z) < 0.1]
    assert len(floor) > 0.1 * len(z) and np.sqrt(np.mean(floor ** 2)) < 0.02      # floor is flat at z = 0
    assert abs(np.percentile(z, 99.5) - room.h) < 0.08                            # ceiling height is right
    lo, hi = np.percentile(out.xyz[:, :2], [0.5, 99.5], axis=0)
    assert abs((hi - lo)[0] - room.lx) < 0.1 and abs((hi - lo)[1] - room.ly) < 0.1  # long side along X, in metres
    assert np.allclose(out.sorted_images()[0].center()[:2], 0, atol=1e-6)         # origin under the first camera
    assert not tf["warnings"]


def test_reference_alignment_carries_the_frame_over():
    _, gt = _tiny_model()
    s, R, t = synth.random_similarity(seed=21)
    other = align.apply_similarity(gt, s, R, t)
    # perturb two cameras badly: the robust fit must ignore them
    ims = other.sorted_images()
    for im in ims[:2]:
        im.tvec = im.tvec + 5.0
    tf = align.align_to_reference(other, gt)
    back = align.apply_similarity(other, tf["s"], tf["R"], tf["t"])
    err = np.linalg.norm(back.centers()[2:] - gt.centers()[2:], axis=1)
    assert err.max() < 1e-3 and tf["diagnostics"]["inliers"] == len(ims) - 2


def test_marker_gives_metric_scale(run):
    d = run["transform"]["diagnostics"]
    assert run["transform"]["scale_source"] == "marker"
    assert abs(d["model_scale_error_pct"] - 3.0) < 0.5        # the draft was handed over 3% too big
    assert abs(d["camera_height_m"] - 1.45) < 0.03            # and comes out at the true camera height
    assert d["marker_check"]["tilt_from_floor_deg"] < 1.0     # the marker lies flat on the found floor


# ------------------------------------------------------------------ linter

def test_linter_finds_the_injected_defects(run):
    clip = run["lint"]["clips"][0]

    def overlaps(kind, a, b):
        return any(i["type"] == kind and i["t_start"] < b and i["t_end"] > a for i in clip["issues"])

    assert overlaps("too_fast", 30.0, 31.2), clip["issues"]   # the whip-pan
    assert overlaps("blurry", 60.0, 61.5), clip["issues"]     # the motion-blurred stretch
    fast = [i for i in clip["issues"] if i["type"] == "too_fast" and i["t_start"] < 31.2 and i["t_end"] > 30.0]
    assert fast[0]["severity"] == "fail"
    # the rest of the walk is mostly clean (close-ups of magnified synthetic textures are honestly soft)
    assert clip["fractions"]["blurry"] < 0.08 and clip["fractions"]["very_fast"] < 0.03
    assert clip["verdict"] == "PASS"


def test_keyframes_follow_parallax_and_avoid_blur(run):
    clip = run["lint"]["clips"][0]
    kfs = clip["keyframe_list"]
    assert 150 < len(kfs) < 400
    clean = [k for k in kfs if not k["weak"]]
    in_blur = [k for k in clean if 60.2 < k["t"] < 61.3]
    assert not in_blur                                         # no clean keyframe was taken from the blurred span
    assert all(os.path.isfile(os.path.join(run["work"], "keyframes", k["name"])) for k in kfs)
    weak_file = open(os.path.join(run["work"], "weak_keyframes.txt")).read().split()
    assert set(weak_file) >= {k["name"] for k in kfs if k["weak"]}


# ------------------------------------------------------------------ audit and the recapture loop

def test_ticket_points_into_the_unvisited_alcove(run):
    before = run["audit_before"]
    assert before["summary"]["verdict"] == "RECAPTURE"
    weak = [it for it in before["items"] if it["kind"] == "weak_surface"]
    assert weak
    # the alcove, moved from the room's own frame into the aligned frame (origin under the first camera)
    x0, y0, x1, y1 = run["room"]["alcove"]
    ox, oy = 0.6, 0.6
    big = max(weak, key=lambda it: it["area_m2"])
    tx, ty = big["target"][0] + ox, big["target"][1] + oy
    assert x0 - 0.3 <= tx <= x1 + 0.3 and y0 - 0.4 <= ty <= y1 + 0.3, (tx, ty)
    assert "side-step" in big["instruction"] or "sweep" in big["instruction"]


def test_recapture_clip_closes_the_loop(run):
    b, a = run["audit_before"]["summary"], run["audit_after"]["summary"]
    assert a["verdict"] == "READY" and a["blocking_items"] == 0
    assert a["by_band"]["mid_pct"] > b["by_band"]["mid_pct"] + 2
    assert a["walk_view_quality_pct"] > 98


def test_frontier_is_reported_where_the_scan_stops():
    # a 4 x 3 m room scanned only in its left 2.5 m: the floor just ends, with no wall there
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(0, 2.5, 40000), rng.uniform(0, 3, 40000), rng.normal(0, 0.005, 40000)]
    walls = []
    for x, y0, y1 in ((0.0, 0, 3),):
        walls.append(np.c_[np.full(8000, x), rng.uniform(y0, y1, 8000), rng.uniform(0, 2.5, 8000)])
    for y in (0.0, 3.0):
        walls.append(np.c_[rng.uniform(0, 2.5, 8000), np.full(8000, y), rng.uniform(0, 2.5, 8000)])
    pts = np.concatenate([floor] + walls)
    centers = np.c_[np.linspace(0.6, 2.0, 30), np.full(30, 1.5), np.full(30, 1.45)]
    grid = navgrid.build(pts, centers)
    fronts, closure = navgrid.frontiers(grid)
    assert len(fronts) == 1
    f = fronts[0]
    assert abs(f["centre"][0] - 2.5) < 0.15 and abs(f["length_m"] - 3.0) < 0.5
    assert f["direction"][0] > 0.9                         # pointing out into the unscanned side
    assert closure["closure_pct"] < 80


# ------------------------------------------------------------------ nav grid

def test_nav_grid_matches_the_room(run):
    nav = run["nav"]
    assert abs(nav["floor_area_m2"] - 19.0) < 1.2          # 20 m2 less the table and partitions
    assert abs(nav["ceiling_height_m"] - 2.6) < 0.1
    meta = json.load(open(os.path.join(run["work"], "package", "nav.json")))
    assert meta["width"] * meta["cell"] > 5 and meta["spawn"]["eye_height"] == 1.6


def test_walls_block_walking_and_sight():
    room, gt = _tiny_model()
    c = gt.centers()
    grid = navgrid.build(gt.xyz, c)
    assert grid.walk[tuple(grid.to_cell(c[0, :2])[::-1])]             # the walker starts on walkable floor
    inside_table = np.array([(room.table[0] + room.table[2]) / 2, (room.table[1] + room.table[3]) / 2])
    assert not grid.walk[tuple(grid.to_cell(inside_table)[::-1])]      # not on the table
    outside = np.array([-0.3, 2.0])
    assert not grid.walk[tuple(grid.to_cell(outside)[::-1])]           # not through the wall
    a = np.array([3.0, 2.2])
    b = np.array([4.2, 3.4])                                           # inside the alcove, behind the partition
    assert not grid.line_of_sight(a + [-0.2, 0.9], b)                  # the partition blocks the direct line


# ------------------------------------------------------------------ pairs and clean-up

def test_pose_guided_pairs_are_sparse_but_connected():
    _, gt = _tiny_model()
    n = len(gt.images)
    pr = pairs.pose_guided_pairs(gt, k=10)
    names = [im.name for im in gt.sorted_images()]
    assert len(pr) < 0.5 * n * (n - 1) / 2                             # far fewer than exhaustive
    have = set(pr)
    assert all((names[i], names[i + 1]) in have for i in range(n - 1))  # every neighbour in time is matched
    # the walk is two laps of the same room: some pairs must link frames far apart in time
    idx = {nm: i for i, nm in enumerate(names)}
    assert any(abs(idx[a] - idx[b]) > n // 3 for a, b in pr)


def test_splat_clean_drops_debris(tmp_path):
    room = synth.Room()
    src = str(tmp_path / "in.ply")
    n = synth.write_splat_ply(room, src, spacing=0.1)
    v = cio.read_ply_vertices(src).copy()
    v["x"][0] = np.nan            # broken
    v["x"][1] = 40.0              # far outside the room
    v["opacity"][2] = -12.0       # invisible
    v["scale_0"][3] = np.log(5.0)  # a five-metre blob
    with open(src, "rb") as f:
        head = f.read().split(b"end_header\n")[0] + b"end_header\n"
    with open(src, "wb") as f:
        f.write(head + v.tobytes())
    lo, hi = np.array([-1, -1, -0.3]), np.array([room.lx + 1, room.ly + 1, room.h + 1.0])
    stats = splat.clean(src, str(tmp_path / "out.ply"), lo, hi)
    assert stats == dict(input=n, kept=n - 4, not_finite=1, outside_room=1, transparent=1, oversized=1)
    assert len(cio.read_ply_vertices(str(tmp_path / "out.ply"))) == n - 4


# ------------------------------------------------------------------ report

def test_report_and_scene_are_published(run):
    scene = os.path.join(run["work"], "scene")
    man = json.load(open(os.path.join(scene, "manifest.json")))
    rep = json.load(open(os.path.join(scene, "report.json")))
    for key in ("splat", "nav", "report"):
        assert os.path.isfile(os.path.join(scene, man[key]))
    assert rep["capture"]["keyframes"] == run["lint"]["total_keyframes"]
    assert rep["coverage"]["verdict"] == "READY"
    assert rep["asset"]["gaussians"] > 50_000 and rep["asset"]["size_mb"] > 0
    assert rep["space"]["floor_area_m2"] == run["nav"]["floor_area_m2"]
