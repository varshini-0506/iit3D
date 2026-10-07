"""Tidy a trained 3DGS PLY before it is compressed and published.

Training leaves debris: Gaussians far outside the room (seen through windows
and doorways), near-transparent ones that only cost download size, and the odd
metre-wide blob. Because the splat is trained in the metric, gravity-aligned
frame, the clean-up is three plain rules in metres.
"""

from __future__ import annotations

import numpy as np

from . import colmap_io as cio


def room_bounds(xyz: np.ndarray, centers: np.ndarray, margin: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box around the scanned space: robust point extent plus every camera, plus a margin."""
    lo, hi = np.percentile(xyz, [0.5, 99.5], axis=0)
    lo, hi = np.minimum(lo, centers.min(0)), np.maximum(hi, centers.max(0))
    lo, hi = lo - margin, hi + margin
    lo[2] = min(lo[2], -0.3)
    return lo, hi


def clean(in_ply: str, out_ply: str, lo: np.ndarray, hi: np.ndarray, min_opacity: float = 0.02,
          max_scale_m: float = 1.5) -> dict:
    """Write a copy of the splat keeping only finite, visible, sensibly sized Gaussians inside [lo, hi]."""
    v = cio.read_ply_vertices(in_ply)
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    for k in names:
        finite &= np.isfinite(v[k])
    inside = ((xyz >= lo) & (xyz <= hi)).all(axis=1)
    visible = np.ones(len(v), bool)
    if "opacity" in names:
        visible = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64))) >= min_opacity
    sane = np.ones(len(v), bool)
    if "scale_0" in names:
        smax = np.max(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float64), axis=1)
        sane = np.exp(np.clip(smax, -50, 50)) <= max_scale_m
    keep = finite & inside & visible & sane
    out = np.ascontiguousarray(v[keep])
    type_names = {"f4": "float", "f8": "double", "u1": "uchar", "i1": "char", "i2": "short", "u2": "ushort",
                  "i4": "int", "u4": "uint"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(out)}"]
    header += [f"property {type_names[out.dtype[k].str[1:]]} {k}" for k in names]
    header.append("end_header")
    with open(out_ply, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(out.astype(out.dtype.newbyteorder("<")).tobytes())
    return dict(input=int(len(v)), kept=int(keep.sum()), not_finite=int((~finite).sum()),
                outside_room=int((finite & ~inside).sum()), transparent=int((finite & inside & ~visible).sum()),
                oversized=int((finite & inside & visible & ~sane).sum()))
