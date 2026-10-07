"""Walkable grid and floor plan from an aligned point cloud.

One 2D grid (5 cm cells by default) is derived from the metric, gravity-aligned
points and reused everywhere:

    viewer      collision for first-person walking, and the minimap
    audit       where a person can stand, and where the scan has open frontiers
    Round 2     the map that path planning runs on

A cell is *free* when the floor was observed there, nothing occupies the body
height band above it, and it connects to where the camera actually walked. The
capture path itself is always carved free: a person demonstrably fitted there.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class NavConfig:
    cell: float = 0.05           # metres per cell
    floor_band: float = 0.10     # |z| below this is floor
    body_lo: float = 0.15        # obstacles are points between body_lo and body_hi
    body_hi: float = 1.90
    wall_lo: float = 1.20        # obstacles that reach this high are drawn as walls
    close_radius: float = 0.12   # fills pin-holes in the observed floor
    carve_radius: float = 0.20   # free corridor around the capture path
    body_radius: float = 0.18    # the walker's radius, for the eroded walk mask
    margin: float = 0.5
    min_frontier: float = 0.4    # shortest open boundary worth reporting, metres
    eye_height: float = 1.6


@dataclass
class NavGrid:
    cell: float
    x0: float
    y0: float
    floor: np.ndarray
    occ: np.ndarray
    wall: np.ndarray
    free: np.ndarray
    walk: np.ndarray
    path: np.ndarray
    spawn: dict
    stats: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape

    def to_cell(self, xy: np.ndarray) -> np.ndarray:
        """World (x, y) -> integer (col, row)."""
        return np.floor((np.asarray(xy, float) - [self.x0, self.y0]) / self.cell).astype(np.int64)

    def to_world(self, cr: np.ndarray) -> np.ndarray:
        """(col, row) -> world (x, y) at the cell centre."""
        return (np.asarray(cr, float) + 0.5) * self.cell + [self.x0, self.y0]

    def inside(self, cr: np.ndarray) -> np.ndarray:
        h, w = self.shape
        cr = np.asarray(cr)
        return (cr[..., 0] >= 0) & (cr[..., 0] < w) & (cr[..., 1] >= 0) & (cr[..., 1] < h)

    def line_of_sight(self, a_xy, b_xy, ignore_near_b: float = 0.3) -> bool:
        """True if the straight floor-plan segment a -> b crosses no occupied cell (except right at b)."""
        a, b = np.asarray(a_xy, float), np.asarray(b_xy, float)
        dist = float(np.linalg.norm(b - a))
        n = max(2, int(dist / (self.cell * 0.5)))
        ts = np.linspace(0, 1, n)
        pts = a + (b - a) * ts[:, None]
        pts = pts[np.linalg.norm(pts - b, axis=1) > ignore_near_b]
        if len(pts) == 0:
            return True
        cr = self.to_cell(pts)
        ok = self.inside(cr)
        return not self.occ[cr[ok, 1], cr[ok, 0]].any()


def _disk(radius_cells: float) -> np.ndarray:
    r = max(1, int(np.ceil(radius_cells)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= radius_cells * radius_cells + 1e-9


def _count(xy: np.ndarray, x0: float, y0: float, cell: float, w: int, h: int) -> np.ndarray:
    c = np.floor((xy - [x0, y0]) / cell).astype(np.int64)
    ok = (c[:, 0] >= 0) & (c[:, 0] < w) & (c[:, 1] >= 0) & (c[:, 1] < h)
    return np.bincount(c[ok, 1] * w + c[ok, 0], minlength=w * h).reshape(h, w)


def _threshold(counts: np.ndarray) -> int:
    nz = counts[counts > 0]
    if len(nz) == 0:
        return 1
    med = float(np.median(nz))
    return 1 if med < 3 else max(2, int(round(0.1 * med)))


def build(xyz: np.ndarray, centers: np.ndarray, forward0: np.ndarray | None = None, cfg: NavConfig | None = None) -> NavGrid:
    """Build the grid from aligned points (metres, Z up, floor at 0) and camera centres."""
    cfg = cfg or NavConfig()
    cell = cfg.cell
    lo, hi = np.percentile(xyz[:, :2], [1, 99], axis=0)
    lo = np.minimum(lo, centers[:, :2].min(0)) - cfg.margin
    hi = np.maximum(hi, centers[:, :2].max(0)) + cfg.margin
    w, h = int(np.ceil((hi[0] - lo[0]) / cell)), int(np.ceil((hi[1] - lo[1]) / cell))
    if w * h > 6000 * 6000:
        raise ValueError(f"Grid of {w}x{h} cells is implausibly large; is the model in metres?")
    x0, y0 = float(lo[0]), float(lo[1])
    z = xyz[:, 2]

    floor_c = _count(xyz[np.abs(z) < cfg.floor_band, :2], x0, y0, cell, w, h)
    body_c = _count(xyz[(z > cfg.body_lo) & (z < cfg.body_hi), :2], x0, y0, cell, w, h)
    high_c = _count(xyz[(z > cfg.wall_lo) & (z < cfg.body_hi), :2], x0, y0, cell, w, h)
    occ = body_c >= _threshold(body_c)
    occ = ndi.binary_closing(occ, structure=np.ones((3, 3), bool))  # make thin walls continuous
    wall = occ & ndi.binary_dilation(high_c >= _threshold(high_c), structure=np.ones((3, 3), bool))
    floor = ndi.binary_closing(floor_c > 0, structure=_disk(cfg.close_radius / cell))

    # the capture path is free space by construction
    carved = np.zeros((h, w), bool)
    pc = np.floor((centers[:, :2] - [x0, y0]) / cell).astype(np.int64)
    for a, b in zip(pc[:-1], pc[1:]):
        if np.linalg.norm((b - a) * cell) < 1.5:  # do not carve across jumps between clips
            n = int(max(abs(b - a).max(), 1))
            for k in range(n + 1):
                c = np.round(a + (b - a) * k / n).astype(int)
                if 0 <= c[0] < w and 0 <= c[1] < h:
                    carved[c[1], c[0]] = True
    ok = (pc[:, 0] >= 0) & (pc[:, 0] < w) & (pc[:, 1] >= 0) & (pc[:, 1] < h)
    carved[pc[ok, 1], pc[ok, 0]] = True
    core = carved.copy()
    carved = ndi.binary_dilation(carved, structure=_disk(cfg.carve_radius / cell))
    occ &= ~carved
    wall &= occ

    free = (floor | carved) & ~occ
    lab, _ = ndi.label(free, structure=np.ones((3, 3), bool))
    keep = np.unique(lab[core])
    free = np.isin(lab, keep[keep > 0])

    walk = ndi.binary_erosion(free, structure=_disk(cfg.body_radius / cell), border_value=0) | (core & free)
    walk |= ndi.binary_dilation(core, structure=_disk(0.08 / cell)) & free
    lab, _ = ndi.label(walk, structure=np.ones((3, 3), bool))
    keep = np.unique(lab[core])
    walk = np.isin(lab, keep[keep > 0])

    if forward0 is None:
        forward0 = np.array([1.0, 0.0])
    spawn = dict(x=float(centers[0, 0]), y=float(centers[0, 1]),
                 yaw=float(np.arctan2(forward0[1], forward0[0])), eye_height=cfg.eye_height)
    step = np.linalg.norm(np.diff(centers[:, :2], axis=0), axis=1)
    grid = NavGrid(cell, x0, y0, floor, occ, wall, free, walk, centers[:, :2].copy(), spawn)
    grid.stats = dict(
        floor_area_m2=round(float(free.sum() * cell * cell), 2),
        walkable_area_m2=round(float(walk.sum() * cell * cell), 2),
        extent_m=[round(float(w * cell), 2), round(float(h * cell), 2)],
        ceiling_height_m=round(float(np.percentile(z, 98)), 2),
        capture_path_m=round(float(step[step < 1.5].sum()), 2),
    )
    return grid


# --------------------------------------------------------------------------
# frontiers: where the scan stops without a wall
# --------------------------------------------------------------------------

def frontiers(grid: NavGrid, cfg: NavConfig | None = None) -> tuple[list[dict], dict]:
    """Open boundaries between observed free space and never-observed space.

    A finished indoor scan has its free floor enclosed by obstacles. Wherever
    free space instead borders *unknown* cells, the capture ended early: behind
    a sofa, around a corner, through a doorway. This is frontier detection from
    robot exploration, used here to tell a person where to walk next.
    """
    cfg = cfg or NavConfig()
    free, occ = grid.free, grid.occ
    unknown = ~free & ~occ & ~grid.floor
    k4 = ndi.generate_binary_structure(2, 1)
    near_unknown = ndi.binary_dilation(unknown, structure=k4)
    fr = free & near_unknown
    boundary = free & ndi.binary_dilation(~free, structure=k4)
    lab, n = ndi.label(ndi.binary_dilation(fr, structure=np.ones((3, 3), bool)), structure=np.ones((3, 3), bool))
    lab = np.where(fr, lab, 0)
    out = []
    for i in range(1, n + 1):
        rows, cols = np.nonzero(lab == i)
        length = len(rows) * grid.cell
        if length < cfg.min_frontier:
            continue
        # direction from known into unknown: average offset to unknown neighbours
        d = np.zeros(2)
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = rows + dr, cols + dc
            ok = (rr >= 0) & (rr < free.shape[0]) & (cc >= 0) & (cc < free.shape[1])
            d += np.array([dc, dr]) * unknown[rr[ok], cc[ok]].sum()
        nd = np.linalg.norm(d)
        centre = grid.to_world(np.array([cols.mean(), rows.mean()]) - 0.5)
        out.append(dict(length_m=round(float(length), 2), centre=[float(centre[0]), float(centre[1])],
                        direction=(d / nd).tolist() if nd > 0 else [1.0, 0.0],
                        cells=np.stack([cols, rows], axis=1)))
    out.sort(key=lambda f: -f["length_m"])
    total_b = float(boundary.sum() * grid.cell)
    open_len = float(sum(f["length_m"] for f in out))
    summary = dict(boundary_m=round(total_b, 2), open_m=round(open_len, 2),
                   closure_pct=round(100.0 * (1 - open_len / total_b), 1) if total_b > 0 else 0.0)
    return out, summary


# --------------------------------------------------------------------------
# drawing and saving
# --------------------------------------------------------------------------

def render_floorplan(grid: NavGrid, scale: int = 6) -> np.ndarray:
    """BGR floor-plan image, +Y up, with a 1 m scale bar."""
    h, w = grid.shape
    img = np.full((h, w, 3), 236, np.uint8)          # never observed
    img[grid.free] = (255, 255, 255)                  # free floor
    img[grid.occ & ~grid.wall] = (176, 178, 180)      # furniture
    img[grid.wall] = (42, 44, 44)                     # walls and tall things
    img = cv2.resize(img[::-1], (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    pts = (grid.path - [grid.x0, grid.y0]) / grid.cell * scale
    pts[:, 1] = h * scale - pts[:, 1]
    seg = np.linalg.norm(np.diff(grid.path, axis=0), axis=1) < 1.5
    for a, b, ok in zip(pts[:-1], pts[1:], seg):
        if ok:
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (165, 95, 24), 1, cv2.LINE_AA)
    bar = int(round(1.0 / grid.cell * scale))
    y = img.shape[0] - 14
    cv2.line(img, (12, y), (12 + bar, y), (0, 0, 0), 2)
    cv2.putText(img, "1 m", (12, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def save(grid: NavGrid, out_dir: str, extra: dict | None = None) -> dict:
    """Write nav.png (packed masks), floorplan.png and nav.json."""
    os.makedirs(out_dir, exist_ok=True)
    h, w = grid.shape
    packed = np.zeros((h, w, 3), np.uint8)  # stored B, G, R; row 0 is the lowest y
    packed[..., 2] = grid.walk * 255   # R: a walker's centre may be here
    packed[..., 1] = grid.wall * 255   # G: wall or tall obstacle
    packed[grid.free, 0] = 128         # B: 128 = free floor, 255 = any obstacle, 0 = never observed
    packed[grid.occ, 0] = 255
    cv2.imwrite(os.path.join(out_dir, "nav.png"), packed)
    cv2.imwrite(os.path.join(out_dir, "floorplan.png"), render_floorplan(grid))
    keep = np.ones(len(grid.path), bool)
    if len(grid.path) > 2:  # thin the path for the minimap
        keep[1:-1] = np.linalg.norm(np.diff(grid.path, axis=0), axis=1)[:-1] > 0.02
    meta = dict(cell=grid.cell, x0=grid.x0, y0=grid.y0, width=w, height=h, image="nav.png",
                channels=dict(R="walk", G="wall", B="128 free floor, 255 obstacle, 0 unknown"),
                spawn=grid.spawn, path=np.round(grid.path[keep], 3).tolist(), stats=grid.stats)
    if extra:
        meta.update(extra)
    with open(os.path.join(out_dir, "nav.json"), "w") as f:
        json.dump(meta, f)
    return meta
