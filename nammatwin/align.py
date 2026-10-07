"""Put a reconstruction into a canonical building frame.

SfM and feed-forward models return scenes in an arbitrary frame: tilted,
rotated, and often in arbitrary units. Everything downstream (eye height in the
viewer, the floor plan, path lengths in Round 2, AR in Round 3) wants:

    Z up, floor at z = 0          gravity alignment from a RANSAC floor plane
    walls parallel to X and Y     Manhattan alignment by histogram entropy
    1 unit = 1 metre              from a metric model, a printed marker, or camera height
    origin under the first camera so the viewer spawns where the walk began

The result is one similarity transform X' = s * R @ X + t, applied to the
camera poses and points so that the splat is *trained* in the canonical frame.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import colmap_io as cio
from .geometry import fit_plane, rot_z, rotation_between, robust_extent, robust_umeyama


# --------------------------------------------------------------------------
# applying a similarity to a model
# --------------------------------------------------------------------------

def apply_similarity(model: cio.Model, s: float, R: np.ndarray, t: np.ndarray) -> cio.Model:
    """Return a copy of the model with X' = s R X + t. Poses stay rigid."""
    images = {}
    for iid, im in model.images.items():
        Rn = im.R() @ R.T
        tn = s * im.tvec - Rn @ t
        images[iid] = cio.Image(im.id, cio.rotmat2qvec(Rn), tn, im.camera_id, im.name, im.xys, im.point3D_ids)
    xyz = s * model.xyz @ R.T + t if len(model.xyz) else model.xyz
    return cio.Model(model.cameras, images, xyz, model.rgb, model.point_ids, model.errors, model.tracks)


def compose(s2, R2, t2, s1, R1, t1):
    """Similarity equal to applying (s1, R1, t1) first and then (s2, R2, t2)."""
    return s2 * s1, R2 @ R1, s2 * R2 @ t1 + t2


# --------------------------------------------------------------------------
# gravity: find the floor
# --------------------------------------------------------------------------

def camera_up_prior(model: cio.Model) -> np.ndarray:
    """Average 'up' of the cameras. Phones are held roughly upright, so this is a good first guess."""
    ups = np.array([-im.R()[1, :] for im in model.images.values()])  # world direction of camera -y
    up = ups.mean(0)
    n = np.linalg.norm(up)
    if n < 0.2:
        raise ValueError("Camera orientations are too scattered to guess which way is up")
    return up / n


def find_floor(xyz: np.ndarray, centers: np.ndarray, up_prior: np.ndarray, iters: int = 600,
               max_tilt_deg: float = 30.0, seed: int = 0) -> dict:
    """RANSAC for the floor: the lowest large near-horizontal plane with the cameras above it."""
    rng = np.random.default_rng(seed)
    tau = 0.006 * robust_extent(xyz)  # inlier distance, scale-free
    h = xyz @ up_prior
    low = np.flatnonzero(h <= np.percentile(h, 45))  # the floor is in the lower part of the scene
    if len(low) < 50:
        low = np.arange(len(xyz))
    sub = xyz if len(xyz) <= 60000 else xyz[rng.choice(len(xyz), 60000, replace=False)]
    cos_max = np.cos(np.radians(max_tilt_deg))
    cands = []
    for _ in range(iters):
        p = xyz[rng.choice(low, 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n /= ln
        if n @ up_prior < 0:
            n = -n
        if n @ up_prior < cos_max:
            continue
        d = -n @ p[0]
        if np.mean(centers @ n + d > 0) < 0.9:  # cameras must be above the floor (rejects ceilings)
            continue
        count = int((np.abs(sub @ n + d) < tau).sum())
        cands.append((count, n, d))
    if not cands:
        raise ValueError("No floor-like plane found. Make sure the capture includes the floor.")
    best = max(c[0] for c in cands)
    # among well-supported planes prefer the lowest one: tables and beds are also horizontal
    strong = [c for c in cands if c[0] >= 0.5 * best]
    count, n, d = max(strong, key=lambda c: np.median(centers @ c[1] + c[2]))
    for _ in range(3):  # least-squares polish on inliers
        inl = np.abs(xyz @ n + d) < tau
        if inl.sum() < 10:
            break
        n2, d2 = fit_plane(xyz[inl])
        if n2 @ n < 0:
            n2, d2 = -n2, -d2
        n, d = n2, d2
    inl = np.abs(xyz @ n + d) < tau
    return dict(normal=n, d=float(d), inlier_ratio=float(inl.mean()),
                rms=float(np.sqrt(np.mean((xyz[inl] @ n + d) ** 2))) if inl.any() else float("nan"),
                tilt_from_prior_deg=float(np.degrees(np.arccos(np.clip(n @ up_prior, -1, 1)))))


# --------------------------------------------------------------------------
# Manhattan: make the walls axis-aligned
# --------------------------------------------------------------------------

def _entropy(vals: np.ndarray, bin_size: float) -> float:
    lo = vals.min()
    nb = max(1, int(np.ceil((vals.max() - lo) / bin_size)) + 1)
    h = np.bincount(((vals - lo) / bin_size).astype(np.int64), minlength=nb).astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log(p)).sum())


def manhattan_yaw(xy: np.ndarray, bin_size: float) -> tuple[float, float]:
    """Yaw (radians, in [0, pi/2)) that makes wall points pile into the fewest x/y bins.

    Returns (yaw, strength). Strength is how much lower the entropy is at the
    best angle than on average; near zero means the room has no dominant axes.
    """
    if len(xy) > 40000:
        xy = xy[np.random.default_rng(0).choice(len(xy), 40000, replace=False)]

    def cost(th):
        c, s = np.cos(th), np.sin(th)
        return _entropy(c * xy[:, 0] - s * xy[:, 1], bin_size) + _entropy(s * xy[:, 0] + c * xy[:, 1], bin_size)

    coarse = np.radians(np.arange(0, 90, 1.0))
    costs = np.array([cost(th) for th in coarse])
    best = coarse[int(np.argmin(costs))]
    fine = best + np.radians(np.arange(-1.0, 1.01, 0.1))
    fcosts = np.array([cost(th) for th in fine])
    yaw = float(fine[int(np.argmin(fcosts))])
    return yaw, float(costs.mean() - fcosts.min())


# --------------------------------------------------------------------------
# full canonicalisation
# --------------------------------------------------------------------------

def canonicalise(model: cio.Model, scale: float | None = None, camera_height: float | None = None) -> dict:
    """Compute the similarity that puts the model in the canonical building frame.

    scale: multiply the input units by this to get metres (None = input is already metric).
    camera_height: if given, set the scale so the median camera sits this high above the floor.
    """
    if len(model.xyz) < 100:
        raise ValueError("Need a point cloud to align (the model has fewer than 100 points)")
    centers = model.centers()
    up0 = camera_up_prior(model)
    floor = find_floor(model.xyz, centers, up0)
    n, d = floor["normal"], floor["d"]

    # 1. gravity: rotate the floor normal onto +Z and drop the floor to z = 0
    R = rotation_between(n, np.array([0, 0, 1.0]))
    t = np.array([0, 0, d])  # after rotation, z' = n.x, and floor satisfies n.x = -d
    pts = model.xyz @ R.T + t
    cam = centers @ R.T + t
    cam_h = float(np.median(cam[:, 2]))

    # 2. scale
    if camera_height is not None:
        s, scale_source = camera_height / cam_h, "camera_height"
    elif scale is not None:
        s, scale_source = float(scale), "given"
    else:
        s, scale_source = 1.0, "input_is_metric"

    # 3. Manhattan yaw from points at wall height
    band = (pts[:, 2] > 0.25 * cam_h) & (pts[:, 2] < 1.4 * cam_h)
    wall_xy = pts[band if band.sum() > 200 else slice(None), :2]
    yaw, strength = manhattan_yaw(wall_xy, bin_size=0.012 * robust_extent(pts))
    Rz = rot_z(yaw)
    pts2 = pts @ Rz.T
    lo, hi = np.percentile(pts2[:, :2], [2, 98], axis=0)
    if (hi - lo)[1] > (hi - lo)[0]:  # long side of the space along X
        Rz = rot_z(np.pi / 2) @ Rz
    cam2 = cam @ Rz.T
    if cam2[0, 0] > np.median(cam2[:, 0]):  # deterministic sign: the walk starts on the low-X side
        Rz = rot_z(np.pi) @ Rz
        cam2 = cam @ Rz.T

    # 4. origin on the floor under the first camera
    R_total = Rz @ R
    t_total = Rz @ t
    t_total = s * t_total
    first = s * cam2[0]
    t_total -= np.array([first[0], first[1], 0.0])

    cam_h_m = cam_h * s
    warnings = []
    if not 0.8 <= cam_h_m <= 2.0:
        warnings.append(f"median camera height is {cam_h_m:.2f} m; a handheld phone is usually 1.2 to 1.6 m, "
                        "so the scale is probably wrong")
    if floor["inlier_ratio"] < 0.05:
        warnings.append("very few points lie on the floor plane; tilt the phone down more during capture")
    if strength < 0.05:
        warnings.append("no dominant wall directions found; axes may not follow the walls")
    return dict(s=float(s), R=R_total, t=t_total, scale_source=scale_source,
                diagnostics=dict(floor_inlier_ratio=round(floor["inlier_ratio"], 4),
                                 floor_rms_m=round(floor["rms"] * s, 4),
                                 up_prior_error_deg=round(floor["tilt_from_prior_deg"], 2),
                                 manhattan_strength=round(strength, 3),
                                 camera_height_m=round(cam_h_m, 3)),
                warnings=warnings)


def align_to_reference(model: cio.Model, ref: cio.Model) -> dict:
    """Similarity that maps `model` onto `ref` using cameras that share an image name.

    Used to carry the metric, gravity-aligned frame of the feed-forward draft
    over to bundle-adjusted COLMAP poses, which are more precise but unscaled.
    """
    by_name = {im.name: im for im in ref.images.values()}
    src, dst, names = [], [], []
    for im in model.images.values():
        if im.name in by_name:
            src.append(im.center()); dst.append(by_name[im.name].center()); names.append(im.name)
    if len(src) < 4:
        raise ValueError(f"Only {len(src)} images in common with the reference; need at least 4")
    src, dst = np.array(src), np.array(dst)
    s, R, t, keep = robust_umeyama(src, dst)
    res = np.linalg.norm((s * src @ R.T + t) - dst, axis=1)
    return dict(s=float(s), R=R, t=t, scale_source="reference_model",
                diagnostics=dict(common_images=len(src), inliers=int(keep.sum()),
                                 trajectory_rmse_m=round(float(np.sqrt(np.mean(res[keep] ** 2))), 4),
                                 trajectory_max_m=round(float(res[keep].max()), 4)),
                warnings=[] if keep.mean() > 0.8 else
                [f"{(~keep).sum()} of {len(keep)} cameras disagree with the reference by a wide margin"])


# --------------------------------------------------------------------------
# intrinsics for a different image size (draft poses -> full-resolution frames)
# --------------------------------------------------------------------------

def rescale_intrinsics(model: cio.Model, new_width: int, new_height: int) -> cio.Model:
    """Re-express intrinsics for full-size images when the model was made from shrunken, centre-cropped copies.

    Feed-forward models resize each image to cover a fixed target size and then
    centre-crop it. Inverting that: k = max(w'/W, h'/H) is the shrink factor and
    the crop offset is ((W k - w') / 2, (H k - h') / 2).
    """
    cams = {}
    for cid, c in model.cameras.items():
        k = max(c.width / new_width, c.height / new_height)
        ox, oy = (new_width * k - c.width) / 2.0, (new_height * k - c.height) / 2.0
        p = c.params.astype(float).copy()
        if c.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            p[0] /= k; p[1] = (p[1] + ox) / k; p[2] = (p[2] + oy) / k
        else:
            p[0] /= k; p[1] /= k; p[2] = (p[2] + ox) / k; p[3] = (p[3] + oy) / k
        cams[cid] = cio.Camera(cid, c.model, int(new_width), int(new_height), p)
    images = {}
    for iid, im in model.images.items():
        xys = im.xys
        if len(xys):
            c = model.cameras[im.camera_id]
            k = max(c.width / new_width, c.height / new_height)
            ox, oy = (new_width * k - c.width) / 2.0, (new_height * k - c.height) / 2.0
            xys = (xys + [ox, oy]) / k
        images[iid] = cio.Image(im.id, im.qvec, im.tvec, im.camera_id, im.name, xys, im.point3D_ids)
    return cio.Model(cams, images, model.xyz, model.rgb, model.point_ids, model.errors, model.tracks)


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def save_outputs(aligned: cio.Model, tf: dict, out_dir: str, prior: dict | None = None) -> None:
    sparse = os.path.join(out_dir, "sparse", "0")
    cio.write_model(aligned, sparse, ".bin")
    cio.write_points_ply(os.path.join(out_dir, "points.ply"), aligned.xyz, aligned.rgb)
    M = np.eye(4)
    M[:3, :3] = tf["s"] * tf["R"]
    M[:3, 3] = tf["t"]
    record = dict(scale=tf["s"], rotation=np.asarray(tf["R"]).tolist(), translation=np.asarray(tf["t"]).tolist(),
                  matrix=M.tolist(), scale_source=tf["scale_source"], diagnostics=tf["diagnostics"],
                  warnings=tf["warnings"], frame="Z up, floor at z=0, walls along X/Y, metres, origin under first camera")
    if prior:
        record["inherited_from_draft"] = prior
    with open(os.path.join(out_dir, "transform.json"), "w") as f:
        json.dump(record, f, indent=1)
    cams = []
    for im in aligned.sorted_images():
        c = aligned.cameras[im.camera_id]
        fx, fy, cx, cy = c.fxfycxcy()
        cams.append(dict(name=im.name, c2w=np.round(im.c2w(), 5).tolist(), fx=fx, fy=fy, cx=cx, cy=cy,
                         w=c.width, h=c.height))
    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump(cams, f)


def _find_images_dir(model_path: str) -> str | None:
    d = cio.find_model_dir(model_path)[0]
    for cand in (os.path.join(model_path, "images"), os.path.join(d, "..", "images"), os.path.join(d, "..", "..", "images")):
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    return None


def _find_transform_json(model_path: str) -> str | None:
    d = cio.find_model_dir(model_path)[0]
    for cand in (os.path.join(model_path, "transform.json"), os.path.join(d, "..", "..", "transform.json"),
                 os.path.join(d, "..", "transform.json")):
        if os.path.isfile(cand):
            return os.path.normpath(cand)
    return None


def run(model_path: str, out_dir: str, scale: float | None = None, camera_height: float | None = None,
        reference: str | None = None, marker_size: float | None = None, marker_mode: str = "auto",
        marker_id: int | None = None, marker_dict: str = "DICT_4X4_50", images_dir: str | None = None,
        drop: str | None = None) -> dict:
    """Align a model and write it to out_dir/sparse/0 with transform.json, cameras.json and points.ply.

    marker_mode "auto":  take the scale from the marker when it was seen well (5+ keyframes, consistent
                         edges); otherwise keep the model's own scale and only report the difference.
    marker_mode "set":   always take the scale from the marker (for SfM output with arbitrary units).
    marker_mode "check": never change the scale; report how far off the marker says it is.
    drop: text file of image names to leave out of the written model (the linter's weak keyframes).
    """
    model = cio.read_model(model_path)
    images_dir = images_dir or _find_images_dir(model_path)
    print(f"[align] {len(model.images)} cameras, {len(model.xyz)} points from {model_path}")
    prior, marker_set, extra_warnings = None, None, []
    if reference:
        tf = align_to_reference(model, cio.read_model(reference))
        tj = _find_transform_json(reference)
        if tj:
            with open(tj) as f:
                prior = {k: v for k, v in json.load(f).items() if k in ("scale_source", "diagnostics")}
    else:
        if marker_size is not None and marker_mode in ("set", "auto"):
            from . import scale_marker
            try:
                est = scale_marker.estimate(model, images_dir, marker_size, marker_dict, marker_id)
            except ValueError as e:
                if marker_mode == "set":
                    raise
                est = None
                extra_warnings.append(f"marker not usable ({e}); kept the model's own scale")
            if est is not None:
                trusted = est["observations"] >= 5 and est["spread_pct"] <= 3.0
                if marker_mode == "set" or trusted:
                    marker_set = est
                    scale, camera_height = est["scale"], None
                else:
                    extra_warnings.append(f"marker seen {est['observations']} time(s) with {est['spread_pct']:.1f}% "
                                          "edge spread: not trusted for scale; kept the model's own scale")
        tf = canonicalise(model, scale=scale, camera_height=camera_height)
        if marker_set is not None:
            tf["scale_source"] = "marker"
            # how far the incoming model's units were from metres, according to the marker
            tf["diagnostics"]["model_scale_error_pct"] = round(100.0 * (1.0 / marker_set["scale"] - 1.0), 2)
    tf["warnings"] += extra_warnings
    aligned = apply_similarity(model, tf["s"], tf["R"], tf["t"])
    if marker_size is not None:
        from . import scale_marker
        try:
            chk = scale_marker.check(aligned, images_dir, marker_size, marker_dict, marker_id)
            frame_source = (prior or {}).get("scale_source") if reference else tf["scale_source"]
            chk["independent"] = frame_source != "marker"  # was the scale set by something other than this marker?
            tf["diagnostics"]["marker_check"] = chk
            print(f"[align] marker: printed {marker_size * 100:.1f} cm, measured {chk['measured_edge_m'] * 100:.2f} cm "
                  f"({chk['scale_error_pct']:+.2f}%), seen in {chk['observations']} keyframes")
            if abs(chk["scale_error_pct"]) > 5:
                tf["warnings"].append(f"marker says the scale is off by {chk['scale_error_pct']:+.1f}%; "
                                      "rerun with --marker-mode set to take the scale from the marker")
        except ValueError as e:
            tf["warnings"].append(f"marker check skipped: {e}")
    if drop and os.path.isfile(drop):
        with open(drop) as f:
            bad = {ln.strip() for ln in f if ln.strip()}
        keep = {iid: im for iid, im in aligned.images.items() if im.name not in bad}
        print(f"[align] left out {len(aligned.images) - len(keep)} weak keyframe(s) listed in {drop}")
        aligned = cio.Model(aligned.cameras, keep, aligned.xyz, aligned.rgb, aligned.point_ids, aligned.errors,
                            aligned.tracks)
    save_outputs(aligned, tf, out_dir, prior)
    for w in tf["warnings"]:
        print(f"[align] WARNING: {w}")
    print(f"[align] scale x{tf['s']:.4f} ({tf['scale_source']}); {tf['diagnostics']}")
    print(f"[align] wrote {os.path.join(out_dir, 'sparse', '0')}")
    return tf
