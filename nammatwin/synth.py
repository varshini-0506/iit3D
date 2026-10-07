"""Synthetic room for tests and for the sample data in tests/sample.

This is NOT a reconstruction of a real place. It is a procedurally textured
box room with furniture, rendered by warping textures through plane
homographies, which gives the CPU-side tools something with known ground truth:

    a two-pass walkthrough video with a deliberately blurred stretch and a too-fast pan
    exact camera poses and a point cloud, handed over in a tilted, scaled frame
    a printed-size ArUco marker on the floor
    an alcove the walker never steps into, so the audit has work to do, and the
      top-up clip that fixes it
    a Gaussian-splat PLY of the room for the web viewer

World frame: Z up, floor at z = 0, metres. Cameras use the OpenCV convention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from . import colmap_io as cio
from .geometry import camera_visibility, rot_z, rotation_between

PPM = 160  # texture pixels per metre


@dataclass
class Plane:
    origin: np.ndarray  # corner
    u: np.ndarray       # unit vector along texture x
    v: np.ndarray       # unit vector along texture y
    wu: float           # extent along u, metres
    wv: float
    tex: np.ndarray     # (wv*PPM, wu*PPM, 3) BGR
    shell: bool = True  # part of the room shell (drawn first) vs furniture

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.u, self.v)

    def corners(self) -> np.ndarray:
        o, a, b = self.origin, self.u * self.wu, self.v * self.wv
        return np.array([o, o + a, o + a + b, o + b])


def _texture(rng: np.random.Generator, wu: float, wv: float, base, busy: float = 1.0) -> np.ndarray:
    w, h = max(8, int(wu * PPM)), max(8, int(wv * PPM))
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.array(base, np.float32)
    low = cv2.resize(rng.normal(0, 1, (max(2, h // 60), max(2, w // 60))).astype(np.float32), (w, h),
                     interpolation=cv2.INTER_CUBIC)
    img += (10 * busy * low)[..., None]
    b = np.array(base, np.float32)
    for _ in range(int(busy * wu * wv * 4)):  # posters, tiles, shelves: things with corners
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        ww, hh = int(rng.integers(12, 90)), int(rng.integers(12, 90))
        col = (0.45 * b + 0.55 * rng.integers(20, 235, 3)).tolist()
        cv2.rectangle(img, (x, y), (min(w - 1, x + ww), min(h - 1, y + hh)), col, -1 if rng.random() < 0.6 else 3)
    for _ in range(int(busy * wu * wv * 1.2)):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(np.clip(p1[0] + rng.integers(-220, 220), 0, w - 1)), int(np.clip(p1[1] + rng.integers(-220, 220), 0, h - 1)))
        cv2.line(img, p1, p2, (0.5 * b + 0.5 * rng.integers(20, 200, 3)).tolist(), int(rng.integers(1, 4)))
    img += rng.normal(0, 5 * busy + 1.5, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _box(rng, x0, y0, x1, y1, h, base) -> list[Plane]:
    """Four sides and a top of an axis-aligned box standing on the floor."""
    up = np.array([0, 0, 1.0])
    out = []
    sides = [(np.array([x0, y0, 0.0]), np.array([1.0, 0, 0]), x1 - x0),
             (np.array([x1, y0, 0.0]), np.array([0, 1.0, 0]), y1 - y0),
             (np.array([x1, y1, 0.0]), np.array([-1.0, 0, 0]), x1 - x0),
             (np.array([x0, y1, 0.0]), np.array([0, -1.0, 0]), y1 - y0)]
    for o, u, wu in sides:
        out.append(Plane(o, u, up, wu, h, _texture(rng, wu, h, base), shell=False))
    out.append(Plane(np.array([x0, y0, h]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), x1 - x0, y1 - y0,
                     _texture(rng, x1 - x0, y1 - y0, base), shell=False))
    return out


class Room:
    def __init__(self, lx: float = 5.0, ly: float = 4.0, h: float = 2.6, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.lx, self.ly, self.h = lx, ly, h
        X, Y, Z = np.eye(3)
        self.planes = [
            Plane(np.zeros(3), X, Y, lx, ly, _texture(rng, lx, ly, (150, 170, 190), 1.2)),               # floor
            Plane(np.array([0, 0, h]), X, Y, lx, ly, _texture(rng, lx, ly, (235, 235, 235), 0.5)),       # ceiling
            Plane(np.zeros(3), X, Z, lx, h, _texture(rng, lx, h, (205, 215, 200), 1.0)),                 # wall y = 0
            Plane(np.array([lx, 0, 0.0]), Y, Z, ly, h, _texture(rng, ly, h, (200, 205, 225), 1.0)),      # wall x = lx
            Plane(np.array([lx, ly, 0.0]), -X, Z, lx, h, _texture(rng, lx, h, (225, 210, 200), 1.0)),    # wall y = ly
            Plane(np.array([0, ly, 0.0]), -Y, Z, ly, h, _texture(rng, ly, h, (210, 225, 215), 1.0)),     # wall x = 0
        ]
        # a blank stretch of wall, the kind of surface that breaks tracking
        self.planes[2].tex[:, int(2.6 * PPM):int(4.6 * PPM)] = (228, 226, 224)
        # furniture: a table, and a partition that makes an alcove in the far corner
        self.table = (1.2, 1.5, 2.2, 2.3)
        self.planes += _box(rng, *self.table, 0.75, (90, 120, 160))
        self.partition = (3.4, 2.6, 3.5, ly)
        self.planes += _box(rng, *self.partition, 2.0, (70, 90, 110))
        self.planes += _box(rng, 4.3, 2.6, lx, 2.7, 2.0, (70, 90, 110))
        self.alcove = (3.5, 2.7, lx, ly)  # x0, y0, x1, y1: entered only through a 0.8 m doorway on its south side
        # printed ArUco marker lying flat on the floor
        self.marker_size = 0.16
        self.marker_origin = np.array([3.5, 1.1, 0.002])
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        px = int(self.marker_size * PPM * 4)
        m = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(d, 0, px), px // 6, px // 6, px // 6, px // 6,
                               cv2.BORDER_CONSTANT, value=255)
        size = self.marker_size * m.shape[0] / px
        off = (size - self.marker_size) / 2
        # v runs along -Y so that the texture is not mirrored when seen from above
        self.marker_plane = Plane(self.marker_origin + [-off, self.marker_size + off, 0], X, -Y, size, size,
                                  cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), shell=False)
        self.marker_ppm = m.shape[0] / size

    # ---------------------------------------------------------------- rendering

    def render(self, K: np.ndarray, c2w: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        W, H = size
        Rwc = c2w[:3, :3].T
        twc = -Rwc @ c2w[:3, 3]
        cam = c2w[:3, 3]
        img = np.zeros((H, W, 3), np.uint8)
        shell = [p for p in self.planes if p.shell]
        furn = [p for p in self.planes if not p.shell and (cam - p.origin) @ p.normal > 0]  # back-face cull
        furn.sort(key=lambda p: -np.linalg.norm(p.corners().mean(0) - cam))
        for p, ppm in [(p, PPM) for p in shell] + [(self.marker_plane, self.marker_ppm)] + [(p, PPM) for p in furn]:
            poly = self._clip_near((p.corners() @ Rwc.T) + twc)
            if len(poly) < 3:
                continue
            uv = (poly @ K.T)
            uv = uv[:, :2] / uv[:, 2:3]
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [np.round(uv).astype(np.int32)], 255)
            if not mask.any():
                continue
            Hm = K @ np.stack([Rwc @ p.u / ppm, Rwc @ p.v / ppm, Rwc @ p.origin + twc], axis=1)
            warped = cv2.warpPerspective(p.tex, Hm, (W, H), flags=cv2.INTER_LINEAR)
            img[mask > 0] = warped[mask > 0]
        return img

    @staticmethod
    def _clip_near(poly: np.ndarray, near: float = 0.05) -> np.ndarray:
        out = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            ia, ib = a[2] >= near, b[2] >= near
            if ia:
                out.append(a)
            if ia != ib:
                t = (near - a[2]) / (b[2] - a[2])
                out.append(a + t * (b - a))
        return np.array(out) if out else np.zeros((0, 3))

    # ---------------------------------------------------------------- sampling

    def sample(self, spacing: float, rng: np.random.Generator, jitter: float = 0.3):
        """Surface samples: positions, colours, and each sample's plane axes (for splats)."""
        P, C, U, V = [], [], [], []
        for p in self.planes + [self.marker_plane]:
            ppm = self.marker_ppm if p is self.marker_plane else PPM
            sp = spacing if p is not self.marker_plane else min(spacing, 0.008)
            a = np.arange(sp / 2, p.wu, sp)
            b = np.arange(sp / 2, p.wv, sp)
            A, B = np.meshgrid(a, b)
            A = A.ravel() + rng.uniform(-jitter, jitter, A.size) * sp
            B = B.ravel() + rng.uniform(-jitter, jitter, B.size) * sp
            pts = p.origin + A[:, None] * p.u + B[:, None] * p.v
            ti = np.clip((A * ppm).astype(int), 0, p.tex.shape[1] - 1)
            tj = np.clip((B * ppm).astype(int), 0, p.tex.shape[0] - 1)
            P.append(pts); C.append(p.tex[tj, ti][:, ::-1])
            U.append(np.repeat(p.u[None], len(pts), 0)); V.append(np.repeat(p.v[None], len(pts), 0))
        return np.concatenate(P), np.concatenate(C).astype(np.uint8), np.concatenate(U), np.concatenate(V)


# --------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------

def look(pos: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    f = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    r = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    d = np.cross(f, r)
    M = np.eye(4)
    M[:3, :3] = np.stack([r, d, f], axis=1)
    M[:3, 3] = pos
    return M


def intrinsics(width: int, height: int, hfov_deg: float = 68.0) -> np.ndarray:
    f = width / 2 / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def _smooth_path(waypoints, speed: float, fps: float, corner: float = 0.5) -> np.ndarray:
    """Constant-speed positions along a polyline whose corners are rounded off."""
    wp = np.asarray(waypoints, float)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    dense = np.concatenate([np.linspace(wp[i], wp[i + 1], max(2, int(seg[i] / 0.01)), endpoint=False)
                            for i in range(len(wp) - 1)] + [wp[-1:]])
    k = max(1, int(corner / 0.01))
    pad = np.concatenate([np.repeat(dense[:1], k, 0), dense, np.repeat(dense[-1:], k, 0)])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(wp.shape[1])], axis=1)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(sm, axis=0), axis=1))])
    n = max(2, int(d[-1] / speed * fps))
    return np.stack([np.interp(np.linspace(0, d[-1], n), d, sm[:, j]) for j in range(wp.shape[1])], axis=1)


def _unwrap_towards(xy: np.ndarray, target: np.ndarray, sign: float = 1.0) -> np.ndarray:
    v = (target - xy) * sign
    return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))


def walk_trajectory(room: Room, fps: float = 10.0, fast_pan: tuple[float, float] | None = (30.0, 31.2)):
    """The SOP's two passes: round the walls looking in, then round the middle looking out.

    The walker never steps into the alcove behind the partition, so the audit
    has something real to find.
    """
    # pass A: perimeter, 0.6 m off the walls, cutting across the mouth of the alcove
    a_xy = _smooth_path([(0.6, 0.6), (4.4, 0.6), (4.4, 2.05), (2.95, 2.05), (2.95, 3.4), (0.6, 3.4), (0.6, 0.6)],
                        speed=0.30, fps=fps)
    a_yaw = _unwrap_towards(a_xy, np.array([2.3, 1.85]))
    # pass B: a loop round the table, looking outward
    ang = np.linspace(0, 2 * np.pi, int(30 * fps), endpoint=False) - np.pi / 2
    c = np.array([1.7, 1.9])
    b_xy = c + np.stack([1.05 * np.cos(ang), 0.95 * np.sin(ang)], axis=1)
    b_yaw = _unwrap_towards(b_xy, c, -1.0)
    # walk from the end of A to the start of B while turning round
    n_tr = int(9 * fps)
    u = np.linspace(0, 1, n_tr)[:, None]
    tr_xy = a_xy[-1] * (1 - u) + b_xy[0] * u
    b_yaw += np.round((a_yaw[-1] + np.pi - b_yaw[0]) / (2 * np.pi)) * 2 * np.pi
    tr_yaw = a_yaw[-1] * (1 - u[:, 0]) + b_yaw[0] * u[:, 0]
    xy = np.concatenate([a_xy, tr_xy, b_xy])
    yaw = np.concatenate([a_yaw, tr_yaw, b_yaw])
    n = len(xy)
    t = np.arange(n) / fps
    yaw = yaw + 0.30 * np.sin(2 * np.pi * t / 14.0)         # sweep a little either side
    pitch = -0.30 + 0.34 * np.sin(2 * np.pi * t / 11.0)     # floor to ceiling
    z = 1.45 + 0.03 * np.sin(2 * np.pi * t * 0.9)
    if fast_pan:
        a, b = fast_pan
        inside = (t >= a) & (t <= b)
        yaw = yaw + np.where(inside, 1.9 * np.sin(np.pi * (t - a) / (b - a)), 0.0)  # whip to one side and back
    return t, [look(np.array([p[0], p[1], h]), y, pi) for p, h, y, pi in zip(xy, z, yaw, pitch)]


def recapture_trajectory(room: Room, fps: float = 10.0):
    """The top-up clip the ticket asks for: start on the old path, step into the alcove, look round it."""
    xy = _smooth_path([(2.95, 2.05), (3.9, 2.1), (3.9, 3.0), (4.4, 3.4), (4.5, 3.1), (4.0, 3.3), (3.9, 2.9)],
                      speed=0.12, fps=fps, corner=0.25)
    n = len(xy)
    t = np.arange(n) / fps
    s = np.linspace(0, 1, n)
    yaw = np.radians(20) + s * np.radians(400) + 0.2 * np.sin(2 * np.pi * t / 9.0)  # turn slowly, all the way round
    pitch = -0.35 + 0.32 * np.sin(2 * np.pi * t / 10.0)
    return t, [look(np.array([p[0], p[1], 1.45]), y, pi) for p, y, pi in zip(xy, yaw, pitch)]


def write_video(room: Room, path: str, t: np.ndarray, poses: list[np.ndarray], size=(640, 360), fps: float = 10.0,
                blur: tuple[float, float] | None = None, seed: int = 1) -> np.ndarray:
    """Render poses to a video, optionally motion-blurring a time span. Returns K."""
    rng = np.random.default_rng(seed)
    K = intrinsics(*size)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not vw.isOpened():
        raise IOError(f"Cannot open video writer for {path}")
    kernel = np.zeros((15, 15), np.float32)
    kernel[7, :] = 1 / 15  # horizontal motion blur
    for ti, c2w in zip(t, poses):
        img = room.render(K, c2w, size)
        if blur and blur[0] <= ti <= blur[1]:
            img = cv2.filter2D(img, -1, kernel)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        vw.write(img)
    vw.release()
    return K


# --------------------------------------------------------------------------
# a stand-in for the draft reconstruction
# --------------------------------------------------------------------------

def draft_model(room: Room, K: np.ndarray, size: tuple[int, int], c2ws: list[np.ndarray], names: list[str],
                spacing: float = 0.04, noise: float = 0.004, seed: int = 3) -> cio.Model:
    """Ground-truth poses plus only the surface that those cameras actually saw.

    A real draft contains nothing for surface no camera saw, so unseen samples
    are dropped: that is what makes the audit's frontier detection testable.
    """
    rng = np.random.default_rng(seed)
    pts, rgb, _, _ = room.sample(spacing, rng)
    seen = np.zeros(len(pts), bool)
    W, H = size
    for c2w in c2ws:
        Rwc = c2w[:3, :3].T
        seen |= camera_visibility(pts, Rwc, -Rwc @ c2w[:3, 3], K[0, 0], K[1, 1], K[0, 2], K[1, 2], W, H, 8.0)
    pts, rgb = pts[seen] + rng.normal(0, noise, (int(seen.sum()), 3)), rgb[seen]
    cam = cio.Camera(1, "PINHOLE", W, H, np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
    images = {}
    for i, (c2w, name) in enumerate(zip(c2ws, names), 1):
        Rwc = c2w[:3, :3].T
        images[i] = cio.Image(i, cio.rotmat2qvec(Rwc), -Rwc @ c2w[:3, 3], 1, name)
    n = len(pts)
    return cio.Model({1: cam}, images, pts, rgb, np.arange(1, n + 1), np.zeros(n), [np.zeros((0, 2), np.int32)] * n)


def random_similarity(seed: int = 11, scale: float | None = None):
    """A tilt, a spin, a shift and a scale: the arbitrary frame a reconstruction comes back in."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    tilt = rotation_between(np.array([0, 0, 1.0]), np.array([0, 0, 1.0]) + 0.35 * axis)
    R = tilt @ rot_z(rng.uniform(0, 2 * np.pi))
    s = float(rng.uniform(0.2, 3.0)) if scale is None else scale
    return s, R, rng.uniform(-5, 5, 3)


# --------------------------------------------------------------------------
# a Gaussian-splat PLY of the room, for the viewer
# --------------------------------------------------------------------------

def write_splat_ply(room: Room, path: str, spacing: float = 0.03, seed: int = 5, transform=None) -> int:
    """Write the room as flat Gaussian discs. `transform` = (s, R, t) moves it into another frame."""
    rng = np.random.default_rng(seed)
    pts, rgb, U, V = room.sample(spacing, rng)
    n = len(pts)
    if transform is not None:
        s, R, t = transform
        pts, U, V, spacing = s * pts @ R.T + t, U @ R.T, V @ R.T, spacing * s
    frames = np.round(np.concatenate([U, V], axis=1), 6)
    uniq, inv = np.unique(frames, axis=0, return_inverse=True)  # one quaternion per plane orientation
    quats = np.array([cio.rotmat2qvec(np.stack([f[:3], f[3:], np.cross(f[:3], f[3:])], axis=1)) for f in uniq])
    q = quats[inv.ravel()]
    sp = np.full(n, spacing)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.zeros(n, dtype=[(k, "<f4") for k in names])
    arr["x"], arr["y"], arr["z"] = pts.T
    dc = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.28209479177387814
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc.T
    arr["opacity"] = 4.0  # logit; sigmoid(4) = 0.98
    arr["scale_0"] = arr["scale_1"] = np.log(sp * 0.62)
    arr["scale_2"] = np.log(sp * 0.08)  # flat discs lying in the surface
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = q.T
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
                 + "".join(f"property float {k}\n" for k in names) + "end_header\n").encode())
        f.write(arr.tobytes())
    return n
