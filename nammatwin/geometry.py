"""Shared geometry helpers: similarity fitting, plane RANSAC, voxel thinning, visibility."""

from __future__ import annotations

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) with dst ~= s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if with_scale and var > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_umeyama(src: np.ndarray, dst: np.ndarray, iters: int = 4, k: float = 3.0):
    """Umeyama with iterative trimming of gross outliers. Returns (s, R, t, inlier_mask)."""
    keep = np.ones(len(src), bool)
    s, R, t = umeyama(src, dst)
    for _ in range(iters):
        res = np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1)
        thr = max(k * np.median(res[keep]), 1e-9)
        new = res <= thr
        if new.sum() < max(4, 0.5 * len(src)) or (new == keep).all():
            break
        keep = new
        s, R, t = umeyama(src[keep], dst[keep])
    return s, R, t, keep


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def fit_plane(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane n.x + d = 0 through pts (n is unit length)."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    return n, float(-n @ c)


def voxel_downsample(xyz: np.ndarray, size: float, rgb: np.ndarray | None = None, max_points: int | None = None,
                     seed: int = 0):
    """Keep one point per voxel (the first seen); optionally cap the count."""
    if len(xyz) == 0:
        return (xyz, rgb) if rgb is not None else xyz
    keys = np.floor(xyz / size).astype(np.int64)
    keys -= keys.min(0)
    dims = keys.max(0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, first = np.unique(flat, return_index=True)
    first.sort()
    if max_points is not None and len(first) > max_points:
        first = np.sort(np.random.default_rng(seed).choice(first, max_points, replace=False))
    return (xyz[first], rgb[first]) if rgb is not None else xyz[first]


def robust_extent(xyz: np.ndarray) -> float:
    """Scene size that ignores stray points: diagonal of the 2-98 percentile box."""
    lo, hi = np.percentile(xyz, [2, 98], axis=0)
    return float(np.linalg.norm(hi - lo))


def camera_visibility(xyz: np.ndarray, R: np.ndarray, t: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                      width: int, height: int, max_range: float, zbuf_px: int = 96,
                      tol_abs: float = 0.06, tol_rel: float = 0.04) -> np.ndarray:
    """Boolean mask of points this camera sees, with a coarse z-buffer for occlusion.

    The point cloud is its own occluder: points are binned into a low-resolution
    depth image and a point counts as visible only if it lies at the front of
    its bin. Coarse bins make this robust to sparse or uneven sampling.
    """
    pc = xyz @ R.T + t
    z = pc[:, 2]
    ok = (z > 0.05) & (z < max_range)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return np.zeros(len(xyz), bool)
    u = fx * pc[idx, 0] / z[idx] + cx
    v = fy * pc[idx, 1] / z[idx] + cy
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    idx, u, v = idx[inside], u[inside], v[inside]
    if len(idx) == 0:
        return np.zeros(len(xyz), bool)
    s = zbuf_px / max(width, height)
    bw, bh = max(1, int(np.ceil(width * s))), max(1, int(np.ceil(height * s)))
    bu = np.minimum((u * s).astype(np.int64), bw - 1)
    bv = np.minimum((v * s).astype(np.int64), bh - 1)
    cell = bv * bw + bu
    zmin = np.full(bw * bh, np.inf)
    np.minimum.at(zmin, cell, z[idx])
    front = z[idx] <= zmin[cell] * (1 + tol_rel) + tol_abs
    out = np.zeros(len(xyz), bool)
    out[idx[front]] = True
    return out
