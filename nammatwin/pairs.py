"""Pose-guided image pairs for structure-from-motion.

Bundle-adjusted COLMAP poses are more precise than feed-forward ones, but
finding which images overlap is COLMAP's slow part: exhaustive matching is
quadratic, and sequential matching misses loop closures and cannot connect a
recapture clip to the footage it tops up.

The draft reconstruction already knows roughly where every keyframe was. So
each image is paired only with the cameras that stood nearby and looked in a
similar direction, plus its neighbours in time. Matching cost becomes linear
in the number of images, and loop closures and recapture clips are linked for
free.
"""

from __future__ import annotations

import numpy as np

from . import colmap_io as cio


def pose_guided_pairs(model: cio.Model, k: int = 25, max_dist: float = 4.0, max_angle_deg: float = 80.0,
                      sequential: int = 6) -> list[tuple[str, str]]:
    """Pairs of image names worth matching, from an (aligned, metric) draft model."""
    ims = model.sorted_images()
    names = [im.name for im in ims]
    C = np.array([im.center() for im in ims])
    F = np.array([im.R()[2] for im in ims])  # optical axes in world coordinates
    n = len(ims)
    pairs: set[tuple[int, int]] = set()
    cos_max = np.cos(np.radians(max_angle_deg))
    for i in range(n):
        d = np.linalg.norm(C - C[i], axis=1)
        cosang = F @ F[i]
        ok = (d <= max_dist) & (cosang >= cos_max)
        ok[i] = False
        cand = np.flatnonzero(ok)
        score = d[cand] + 2.0 * np.arccos(np.clip(cosang[cand], -1, 1)) / np.pi
        for j in cand[np.argsort(score)[:k]]:
            pairs.add((min(i, int(j)), max(i, int(j))))
        for j in range(i + 1, min(n, i + 1 + sequential)):  # neighbours in time, whatever the draft says
            pairs.add((i, j))
    return [(names[i], names[j]) for i, j in sorted(pairs)]


def write_pairs(pairs: list[tuple[str, str]], path: str) -> None:
    with open(path, "w") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")
