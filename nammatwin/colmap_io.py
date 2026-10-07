"""Minimal, dependency-free reader/writer for COLMAP sparse models.

COLMAP's sparse model (cameras / images / points3D, as .bin or .txt) is the
interchange format of this pipeline: MapAnything exports it, COLMAP produces
it, and gsplat trains from it. Keeping our own small implementation means the
CPU-side tools run anywhere without pycolmap.

Conventions: world-to-camera pose, x_cam = R(qvec) @ X_world + tvec, with the
OpenCV camera frame (x right, y down, z forward). qvec is (w, x, y, z).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np

# model_id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {name: mid for mid, (name, _) in CAMERA_MODELS.items()}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def fxfycxcy(self) -> tuple[float, float, float, float]:
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                          "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            return float(p[0]), float(p[0]), float(p[1]), float(p[2])
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])

    def K(self) -> np.ndarray:
        fx, fy, cx, cy = self.fxfycxcy()
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    def dist_coeffs(self) -> np.ndarray:
        """OpenCV-style (k1, k2, p1, p2) distortion; zeros when pinhole."""
        p = self.params
        if self.model == "SIMPLE_RADIAL":
            return np.array([p[3], 0, 0, 0.0])
        if self.model == "RADIAL":
            return np.array([p[3], p[4], 0, 0.0])
        if self.model in ("OPENCV", "FULL_OPENCV"):
            return np.array([p[4], p[5], p[6], p[7]], dtype=float)
        return np.zeros(4)


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (w, x, y, z)
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def R(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R().T @ self.tvec

    def c2w(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R().T
        M[:3, 3] = self.center()
        return M


@dataclass
class Model:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    xyz: np.ndarray  # (N, 3) float64
    rgb: np.ndarray  # (N, 3) uint8
    point_ids: np.ndarray  # (N,) int64
    errors: np.ndarray  # (N,) float64
    tracks: list[np.ndarray] | None = None  # per point: (L, 2) [image_id, point2D_idx]

    def sorted_images(self) -> list[Image]:
        return sorted(self.images.values(), key=lambda im: im.name)

    def centers(self) -> np.ndarray:
        return np.array([im.center() for im in self.sorted_images()])


# --------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------

def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -q if q[0] < 0 else q


# --------------------------------------------------------------------------
# locating a model on disk
# --------------------------------------------------------------------------

def find_model_dir(path: str) -> tuple[str, str]:
    """Return (dir, ext) for a COLMAP model given the model dir or a parent."""
    for sub in ("", "sparse/0", "sparse", "0"):
        d = os.path.join(path, sub) if sub else path
        for ext in (".bin", ".txt"):
            if os.path.isfile(os.path.join(d, "images" + ext)):
                return d, ext
    raise FileNotFoundError(f"No COLMAP model (images.bin/.txt) found under {path}")


def read_model(path: str) -> Model:
    d, ext = find_model_dir(path)
    if ext == ".bin":
        cams = _read_cameras_bin(os.path.join(d, "cameras.bin"))
        imgs = _read_images_bin(os.path.join(d, "images.bin"))
        pts = _read_points_bin(os.path.join(d, "points3D.bin"))
    else:
        cams = _read_cameras_txt(os.path.join(d, "cameras.txt"))
        imgs = _read_images_txt(os.path.join(d, "images.txt"))
        pts = _read_points_txt(os.path.join(d, "points3D.txt"))
    return Model(cams, imgs, *pts)


def write_model(model: Model, out_dir: str, ext: str = ".bin") -> None:
    os.makedirs(out_dir, exist_ok=True)
    if ext == ".bin":
        _write_cameras_bin(model.cameras, os.path.join(out_dir, "cameras.bin"))
        _write_images_bin(model.images, os.path.join(out_dir, "images.bin"))
        _write_points_bin(model, os.path.join(out_dir, "points3D.bin"))
    else:
        _write_cameras_txt(model.cameras, os.path.join(out_dir, "cameras.txt"))
        _write_images_txt(model.images, os.path.join(out_dir, "images.txt"))
        _write_points_txt(model, os.path.join(out_dir, "points3D.txt"))


# --------------------------------------------------------------------------
# binary
# --------------------------------------------------------------------------

def _rd(f, fmt: str):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def _read_cameras_bin(path: str) -> dict[int, Camera]:
    cams = {}
    with open(path, "rb") as f:
        (n,) = _rd(f, "Q")
        for _ in range(n):
            cid, mid, w, h = _rd(f, "iiQQ")
            name, npar = CAMERA_MODELS[mid]
            params = np.array(_rd(f, "d" * npar))
            cams[cid] = Camera(cid, name, int(w), int(h), params)
    return cams


def _read_images_bin(path: str) -> dict[int, Image]:
    imgs = {}
    with open(path, "rb") as f:
        (n,) = _rd(f, "Q")
        for _ in range(n):
            vals = _rd(f, "idddddddi")
            iid, q, t, cid = vals[0], np.array(vals[1:5]), np.array(vals[5:8]), vals[8]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name += c
            (n2d,) = _rd(f, "Q")
            raw = np.frombuffer(f.read(24 * n2d), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            xys = np.stack([raw["x"], raw["y"]], axis=1) if n2d else np.zeros((0, 2))
            imgs[iid] = Image(iid, q, t, cid, name.decode("utf-8"), xys, raw["id"].astype(np.int64).copy())
    return imgs


def _read_points_bin(path: str):
    if not os.path.isfile(path):
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0, np.int64), np.zeros(0), []
    ids, xyz, rgb, err, tracks = [], [], [], [], []
    with open(path, "rb") as f:
        (n,) = _rd(f, "Q")
        for _ in range(n):
            pid, x, y, z, r, g, b, e = _rd(f, "QdddBBBd")
            (tl,) = _rd(f, "Q")
            tr = np.frombuffer(f.read(8 * tl), dtype="<i4").reshape(-1, 2).copy()
            ids.append(pid); xyz.append((x, y, z)); rgb.append((r, g, b)); err.append(e); tracks.append(tr)
    return (np.array(xyz, dtype=np.float64).reshape(-1, 3), np.array(rgb, dtype=np.uint8).reshape(-1, 3),
            np.array(ids, dtype=np.int64), np.array(err, dtype=np.float64), tracks)


def _write_cameras_bin(cams: dict[int, Camera], path: str) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for c in cams.values():
            f.write(struct.pack("<iiQQ", c.id, CAMERA_MODEL_IDS[c.model], c.width, c.height))
            f.write(struct.pack("<" + "d" * len(c.params), *[float(p) for p in c.params]))


def _write_images_bin(imgs: dict[int, Image], path: str) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for im in imgs.values():
            f.write(struct.pack("<idddddddi", im.id, *[float(v) for v in im.qvec],
                                *[float(v) for v in im.tvec], im.camera_id))
            f.write(im.name.encode("utf-8") + b"\x00")
            n2d = len(im.xys)
            f.write(struct.pack("<Q", n2d))
            if n2d:
                raw = np.zeros(n2d, dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
                raw["x"], raw["y"], raw["id"] = im.xys[:, 0], im.xys[:, 1], im.point3D_ids
                f.write(raw.tobytes())


def _write_points_bin(m: Model, path: str) -> None:
    with open(path, "wb") as f:
        n = len(m.xyz)
        f.write(struct.pack("<Q", n))
        for i in range(n):
            f.write(struct.pack("<QdddBBBd", int(m.point_ids[i]), *[float(v) for v in m.xyz[i]],
                                *[int(v) for v in m.rgb[i]], float(m.errors[i])))
            tr = m.tracks[i] if m.tracks else np.zeros((0, 2), np.int32)
            f.write(struct.pack("<Q", len(tr)))
            if len(tr):
                f.write(np.asarray(tr, dtype="<i4").tobytes())


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------

def _lines(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def _read_cameras_txt(path: str) -> dict[int, Camera]:
    cams = {}
    for line in _lines(path):
        p = line.split()
        cams[int(p[0])] = Camera(int(p[0]), p[1], int(p[2]), int(p[3]), np.array(p[4:], dtype=float))
    return cams


def _read_images_txt(path: str) -> dict[int, Image]:
    imgs = {}
    with open(path, "r") as f:
        raw = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    i = 0
    while i < len(raw):
        if not raw[i].strip():
            i += 1
            continue
        p = raw[i].split()
        iid, q, t, cid, name = int(p[0]), np.array(p[1:5], float), np.array(p[5:8], float), int(p[8]), " ".join(p[9:])
        pts = raw[i + 1].split() if i + 1 < len(raw) else []
        arr = np.array(pts, dtype=float).reshape(-1, 3) if pts else np.zeros((0, 3))
        imgs[iid] = Image(iid, q, t, cid, name, arr[:, :2].copy(), arr[:, 2].astype(np.int64))
        i += 2
    return imgs


def _read_points_txt(path: str):
    if not os.path.isfile(path):
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0, np.int64), np.zeros(0), []
    ids, xyz, rgb, err, tracks = [], [], [], [], []
    for line in _lines(path):
        p = line.split()
        ids.append(int(p[0])); xyz.append([float(v) for v in p[1:4]])
        rgb.append([int(v) for v in p[4:7]]); err.append(float(p[7]))
        tracks.append(np.array(p[8:], dtype=np.int32).reshape(-1, 2))
    return (np.array(xyz, dtype=np.float64).reshape(-1, 3), np.array(rgb, dtype=np.uint8).reshape(-1, 3),
            np.array(ids, dtype=np.int64), np.array(err, dtype=np.float64), tracks)


def _write_cameras_txt(cams: dict[int, Camera], path: str) -> None:
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for c in cams.values():
            f.write(f"{c.id} {c.model} {c.width} {c.height} " + " ".join(repr(float(p)) for p in c.params) + "\n")


def _write_images_txt(imgs: dict[int, Image], path: str) -> None:
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n"
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for im in imgs.values():
            f.write(f"{im.id} " + " ".join(repr(float(v)) for v in im.qvec) + " "
                    + " ".join(repr(float(v)) for v in im.tvec) + f" {im.camera_id} {im.name}\n")
            f.write(" ".join(f"{x:.3f} {y:.3f} {int(pid)}" for (x, y), pid in zip(im.xys, im.point3D_ids)) + "\n")


def _write_points_txt(m: Model, path: str) -> None:
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n"
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i in range(len(m.xyz)):
            tr = m.tracks[i] if m.tracks else np.zeros((0, 2), np.int32)
            f.write(f"{int(m.point_ids[i])} " + " ".join(repr(float(v)) for v in m.xyz[i]) + " "
                    + " ".join(str(int(v)) for v in m.rgb[i]) + f" {float(m.errors[i])!r} "
                    + " ".join(str(int(v)) for v in np.asarray(tr).ravel()) + "\n")


# --------------------------------------------------------------------------
# plain point clouds and 3DGS PLY (positions + opacity only)
# --------------------------------------------------------------------------

_PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}


def read_ply_vertices(path: str) -> np.ndarray:
    """Read the vertex element of a binary/ascii PLY as a structured array."""
    with open(path, "rb") as f:
        header, fmt, n, props, in_vertex = [], None, 0, [], False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of PLY header")
            s = line.decode("ascii", "replace").strip()
            header.append(s)
            tok = s.split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                if tok[1] == "list":
                    raise ValueError("List properties on vertices are not supported")
                props.append((tok[2], _PLY_TYPES[tok[1]]))
            elif tok[0] == "end_header":
                break
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n, ndmin=2)
            out = np.zeros(n, dtype=[(name, t) for name, t in props])
            for i, (name, _) in enumerate(props):
                out[name] = data[:, i]
            return out
        endian = "<" if fmt == "binary_little_endian" else ">"
        dt = np.dtype([(name, endian + t) for name, t in props])
        return np.frombuffer(f.read(dt.itemsize * n), dtype=dt, count=n)


def read_splat_points(path: str, min_opacity: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian centres (and RGB if available) from a 3DGS PLY, dropping faint ones."""
    v = read_ply_vertices(path)
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    keep = np.isfinite(xyz).all(axis=1)
    names = v.dtype.names
    if "opacity" in names:  # stored as a logit in 3DGS PLYs
        keep &= 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64))) >= min_opacity
    if "f_dc_0" in names:
        dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float64)
        rgb = np.clip((0.5 + 0.28209479177387814 * dc) * 255, 0, 255).astype(np.uint8)
    elif "red" in names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 128, np.uint8)
    return xyz[keep], rgb[keep]


def write_points_ply(path: str, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    n = len(xyz)
    rgb = np.full((n, 3), 200, np.uint8) if rgb is None else rgb
    arr = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(arr.tobytes())
