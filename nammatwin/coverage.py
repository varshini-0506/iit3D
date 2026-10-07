"""Coverage audit and recapture ticket.

Runs on the fast draft reconstruction, before any expensive training, and
answers one question: *if we trained now, where would the twin look bad, and
what is the cheapest footage that fixes it?*

Two kinds of problems are found:

    weak surfaces   Surface that was seen, but poorly: from too few keyframes,
                    or only from one direction (no parallax, so depth is a guess
                    and the splat turns to fog when you walk around it).
    frontiers       Places where observed floor borders space that was never
                    observed and no wall closes it: the scan simply stopped.

For every surface sample we count the keyframes that see it (with occlusion)
and measure how spread out their viewing directions are. If S is the sum of n
unit view directions, the effective baseline angle is 2 * acos(|S| / n): 0 when
every view comes from the same direction, growing as views fan out.

Each problem becomes a line on the recapture ticket: where to stand, which way
to face, how to move, and roughly when in the original clip you were nearby.
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import colmap_io as cio
from . import navgrid as ng
from .geometry import camera_visibility, voxel_downsample


@dataclass
class AuditConfig:
    voxel: float = 0.05            # surface sample spacing, metres
    max_points: int = 120_000
    max_range: float = 6.0         # beyond this a view adds little detail
    min_views: int = 4             # fewer sightings than this is weak
    min_baseline_deg: float = 12.0 # narrower fan of view directions than this is weak
    cluster_cell: float = 0.10     # grid used to group weak samples
    min_cluster_m2: float = 0.12   # ignore weak patches smaller than this
    stand_dist: float = 1.4        # preferred distance to stand from a problem
    max_items: int = 8
    ceiling_from: float = 2.0      # samples above this height are reported as 'ceiling', not mapped
    split_tile: float = 2.0        # large weak regions are split into tiles of this size, metres
    ready_surface_pct: float = 90.0  # floor + walls + furniture well observed, percent
    blocking_m2: float = 0.25        # a weak patch at least this big blocks the READY verdict
    ready_closure_pct: float = 90.0


# --------------------------------------------------------------------------
# per-sample observation statistics
# --------------------------------------------------------------------------

def observe(model: cio.Model, cfg: AuditConfig) -> dict:
    pts = voxel_downsample(model.xyz, cfg.voxel, max_points=cfg.max_points)
    n = np.zeros(len(pts), np.int32)
    S = np.zeros((len(pts), 3))
    for im in model.sorted_images():
        cam = model.cameras[im.camera_id]
        fx, fy, cx, cy = cam.fxfycxcy()
        vis = camera_visibility(pts, im.R(), im.tvec, fx, fy, cx, cy, cam.width, cam.height, cfg.max_range)
        if not vis.any():
            continue
        d = im.center() - pts[vis]
        S[vis] += d / np.linalg.norm(d, axis=1, keepdims=True)
        n[vis] += 1
    R = np.where(n > 0, np.linalg.norm(S, axis=1) / np.maximum(n, 1), 1.0)
    baseline = np.degrees(2 * np.arccos(np.clip(R, 0, 1)))
    few = n < cfg.min_views
    narrow = ~few & (baseline < cfg.min_baseline_deg)
    return dict(pts=pts, n=n, baseline=baseline, few=few, narrow=narrow, weak=few | narrow)


# --------------------------------------------------------------------------
# where to stand
# --------------------------------------------------------------------------

def _pick_stand(grid: ng.NavGrid, target_xy: np.ndarray, normal_xy: np.ndarray | None, cfg: AuditConfig,
                r_min: float = 0.8, r_max: float = 2.4) -> tuple[np.ndarray, bool]:
    rows, cols = np.nonzero(grid.walk)
    if len(rows) == 0:
        return target_xy, False
    xy = grid.to_world(np.stack([cols, rows], axis=1))
    off = xy - target_xy
    dist = np.linalg.norm(off, axis=1)
    ring = np.flatnonzero((dist >= r_min) & (dist <= r_max))
    if len(ring):
        score = -0.6 * np.abs(dist[ring] - cfg.stand_dist)
        if normal_xy is not None:
            score += np.abs(off[ring] @ normal_xy) / dist[ring]  # square-on beats grazing
        for j in ring[np.argsort(-score)][:400]:
            if grid.line_of_sight(xy[j], target_xy):
                return xy[j], True
    return xy[int(np.argmin(dist))], False


def _relative_to_start(xy: np.ndarray, grid: ng.NavGrid) -> str:
    yaw = grid.spawn["yaw"]
    f = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-f[1], f[0]])
    d = np.asarray(xy) - [grid.spawn["x"], grid.spawn["y"]]
    a, b = float(d @ f), float(d @ left)
    part_a = f"{abs(a):.1f} m {'ahead of' if a >= 0 else 'behind'}"
    part_b = f"{abs(b):.1f} m to the {'left' if b >= 0 else 'right'}"
    return f"{part_a} and {part_b} of where you started (facing the way you first faced)"


def _nearest_keyframe(xy: np.ndarray, model: cio.Model, lint: dict | None) -> dict | None:
    ims = model.sorted_images()
    c = np.array([im.center()[:2] for im in ims])
    j = int(np.argmin(np.linalg.norm(c - xy, axis=1)))
    info = dict(name=ims[j].name, distance_m=round(float(np.linalg.norm(c[j] - xy)), 2))
    if lint:
        for clip in lint.get("clips", []):
            for kf in clip.get("keyframe_list", []):
                if kf["name"] == ims[j].name:
                    info.update(clip=clip["clip"], t=kf["t"])
    return info


def _when(kf: dict | None) -> str:
    if not kf or "t" not in kf:
        return ""
    t = kf["t"]
    return f" You were about {kf['distance_m']:.1f} m from this spot at {int(t // 60)}:{int(t % 60):02d} of clip {kf['clip']}."


# --------------------------------------------------------------------------
# weak-surface clusters
# --------------------------------------------------------------------------

def weak_clusters(obs: dict, grid: ng.NavGrid, cfg: AuditConfig) -> list[dict]:
    low = obs["pts"][:, 2] <= cfg.ceiling_from  # the ceiling is handled as one item of its own
    pts, weak = obs["pts"], obs["weak"] & low
    c = cfg.cluster_cell
    lo = pts[:, :2].min(0)
    ij = np.floor((pts[:, :2] - lo) / c).astype(np.int64)
    w, h = ij.max(0) + 1
    flat = ij[:, 1] * w + ij[:, 0]
    tot = np.bincount(flat[low], minlength=w * h).reshape(h, w)
    wk = np.bincount(flat[weak], minlength=w * h).reshape(h, w)
    bad = (wk >= 2) & (wk >= 0.5 * tot)
    lab, n = ndi.label(ndi.binary_dilation(bad, structure=np.ones((3, 3), bool)), structure=np.ones((3, 3), bool))
    # a region too big to fix from one spot is cut into tiles, each with its own instruction
    tile = np.floor((pts[:, :2] - lo) / cfg.split_tile).astype(np.int64)
    key = (lab[ij[:, 1], ij[:, 0]].astype(np.int64) * 10007 + tile[:, 1]) * 10007 + tile[:, 0]
    key[lab[ij[:, 1], ij[:, 0]] == 0] = -1
    out = []
    for i in np.unique(key[weak]):
        if i < 0:
            continue
        sel = weak & (key == i)
        area = float(sel.sum() * cfg.voxel * cfg.voxel)
        if area < cfg.min_cluster_m2:
            continue
        p = pts[sel]
        centre = np.median(p, axis=0)
        xy = p[:, :2] - p[:, :2].mean(0)
        evals, evecs = np.linalg.eigh(xy.T @ xy / len(xy))
        elongated = evals[1] > 4 * max(evals[0], 1e-9)
        normal = evecs[:, 0] if elongated and np.ptp(p[:, 2]) > 0.3 else None  # wall-like: face it square-on
        few = int(obs["few"][sel].sum())
        out.append(dict(kind="weak_surface", area_m2=round(area, 2), centre=centre, normal_xy=normal,
                        z_lo=float(np.percentile(p[:, 2], 5)), z_hi=float(np.percentile(p[:, 2], 95)),
                        reason="few_views" if few >= 0.5 * sel.sum() else "narrow_baseline",
                        median_views=float(np.median(obs["n"][sel])),
                        median_baseline_deg=float(np.median(obs["baseline"][sel]))))
    out.sort(key=lambda d: -d["area_m2"])
    return out


def _height_hint(z_lo: float, z_hi: float) -> str:
    mid = 0.5 * (z_lo + z_hi)
    if z_hi < 0.5:
        return "Tilt the phone down: the problem is at floor level."
    if z_lo > 1.8:
        return "Tilt the phone up: the problem is near the ceiling."
    if z_hi - z_lo > 1.5:
        return "Cover it top to bottom: sweep once at chest height, once tilted up, once tilted down."
    return f"Aim at about {max(0.2, mid):.1f} m above the floor."


def build_ticket(obs: dict, grid: ng.NavGrid, fronts: list[dict], model: cio.Model, lint: dict | None,
                 cfg: AuditConfig) -> list[dict]:
    items = []
    for cl in weak_clusters(obs, grid, cfg):
        stand, los = _pick_stand(grid, cl["centre"][:2], cl["normal_xy"], cfg)
        face = cl["centre"][:2] - stand
        kf = _nearest_keyframe(stand, model, lint)
        if cl["reason"] == "few_views":
            why = (f"About {cl['area_m2']:.1f} m2 of surface appears in a median of only "
                   f"{cl['median_views']:.0f} keyframes.")
            how = "Face it and sweep slowly across it for 10 to 15 seconds."
        else:
            why = (f"About {cl['area_m2']:.1f} m2 of surface was only ever seen from one direction "
                   f"(views fan out by {cl['median_baseline_deg']:.0f} degrees), so its depth is a guess.")
            how = "Keep it in frame and side-step about 1 m to each side. Move your feet; do not just pivot."
        items.append(dict(kind="weak_surface", priority=cl["area_m2"], reason=cl["reason"],
                          target=[round(float(v), 2) for v in cl["centre"]],
                          stand=[round(float(v), 2) for v in stand],
                          face_deg=round(float(np.degrees(np.arctan2(face[1], face[0]))), 1),
                          clear_view=bool(los), area_m2=cl["area_m2"], nearest_keyframe=kf,
                          where=_relative_to_start(stand, grid),
                          instruction=f"{why} {how} {_height_hint(cl['z_lo'], cl['z_hi'])}{_when(kf)}"))
    for fr in fronts:
        c, d = np.array(fr["centre"]), np.array(fr["direction"])
        rows, cols = np.nonzero(grid.walk)
        if len(rows) == 0:
            continue
        xy = grid.to_world(np.stack([cols, rows], axis=1))
        stand = xy[int(np.argmin(np.linalg.norm(xy - (c - 0.5 * d), axis=1)))]
        kf = _nearest_keyframe(stand, model, lint)
        items.append(dict(kind="frontier", priority=0.5 * fr["length_m"], reason="unscanned_beyond",
                          target=[round(float(c[0] + 0.6 * d[0]), 2), round(float(c[1] + 0.6 * d[1]), 2), 0.0],
                          stand=[round(float(v), 2) for v in stand],
                          face_deg=round(float(np.degrees(np.arctan2(d[1], d[0]))), 1), clear_view=True,
                          length_m=fr["length_m"], nearest_keyframe=kf, where=_relative_to_start(stand, grid),
                          instruction=(f"The scan stops along a {fr['length_m']:.1f} m edge with no wall behind it: "
                                       "floor beyond here was never seen. Walk through and sweep the area behind, "
                                       "including the floor. If this is a doorway to a room you are leaving out, "
                                       f"close the door or ignore this item.{_when(kf)}")))
    high = obs["pts"][:, 2] > cfg.ceiling_from
    if high.sum() > 50 and obs["weak"][high].mean() > 0.4:
        pct_ok = 100 * (1 - float(obs["weak"][high].mean()))
        start = [grid.spawn["x"], grid.spawn["y"]]
        items.append(dict(kind="ceiling", priority=0.25, reason="ceiling_undersampled",
                          target=[round(start[0], 2), round(start[1], 2), 2.4], stand=[round(v, 2) for v in start],
                          face_deg=round(float(np.degrees(grid.spawn["yaw"])), 1), clear_view=True,
                          nearest_keyframe=None, where="anywhere: repeat your first lap",
                          instruction=(f"Only {pct_ok:.0f}% of the ceiling and upper walls is well observed. "
                                       "Walk one more slow lap of the room with the phone tilted up about 30 degrees, "
                                       "keeping the top of the walls in frame so the new footage ties in.")))
    items.sort(key=lambda d: -d["priority"])
    items = items[:cfg.max_items]
    for i, it in enumerate(items, 1):
        it["id"] = i
        it.pop("priority")
    return items


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def coverage_overlay(obs: dict, grid: ng.NavGrid, cfg: AuditConfig | None = None) -> np.ndarray:
    """RGBA image on the nav grid: green where surfaces are well observed, red where weak.

    A plan view stacks floor, furniture, walls and ceiling on top of each other,
    so the layers are kept apart: walls and furniture are drawn over the floor,
    and the ceiling is left off the map (it is reported as a number instead).
    """
    cfg = cfg or AuditConfig()
    h, w = grid.shape
    z = obs["pts"][:, 2]
    cr = grid.to_cell(obs["pts"][:, :2])
    inside = grid.inside(cr)
    k = np.ones((3, 3))

    def layer(mask):
        ok = inside & mask
        flat = cr[ok, 1] * w + cr[ok, 0]
        tot = ndi.convolve(np.bincount(flat, minlength=w * h).reshape(h, w).astype(float), k, mode="constant")
        wk = ndi.convolve(np.bincount(flat[obs["weak"][ok]], minlength=w * h).reshape(h, w).astype(float), k,
                          mode="constant")
        return np.divide(wk, tot, out=np.zeros_like(wk), where=tot > 0), tot > 0

    f_frac, f_seen = layer(z < 0.1)
    m_frac, m_seen = layer((z >= 0.1) & (z <= cfg.ceiling_from))
    frac = np.where(m_seen, m_frac, f_frac)
    seen = m_seen | f_seen
    good, bad = np.array([29, 158, 117.0]), np.array([226, 75, 74.0])
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = (good[None, None, :] * (1 - frac[..., None]) + bad[None, None, :] * frac[..., None]).astype(np.uint8)
    rgba[..., 3] = np.where(seen, np.where(m_seen, 230, 150), 0)
    return rgba


def render_ticket(obs: dict, grid: ng.NavGrid, items: list[dict], fronts: list[dict], path: str, title: str,
                  cfg: AuditConfig | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = grid.shape
    ext = [grid.x0, grid.x0 + w * grid.cell, grid.y0, grid.y0 + h * grid.cell]
    base = np.full((h, w, 3), 236, np.uint8)
    base[grid.free] = 255
    base[grid.occ & ~grid.wall] = 200
    base[grid.wall] = 70
    fig_w = 10.0
    fig, ax = plt.subplots(figsize=(fig_w, max(4.0, fig_w * h / w)), constrained_layout=True)
    ax.imshow(base, origin="lower", extent=ext, interpolation="nearest")
    ax.imshow(coverage_overlay(obs, grid, cfg), origin="lower", extent=ext, interpolation="nearest")
    seg = np.linalg.norm(np.diff(grid.path, axis=0), axis=1) < 1.5
    for a, b, ok in zip(grid.path[:-1], grid.path[1:], seg):
        if ok:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#185fa5", lw=0.9, alpha=0.7)
    for fr in fronts:
        xy = grid.to_world(fr["cells"])
        ax.scatter(xy[:, 0], xy[:, 1], s=5, color="#7f77dd", marker="s", linewidths=0)
    for it in items:
        sx, sy = it["stand"][:2]
        tx, ty = it["target"][:2]
        if it["kind"] != "ceiling":
            ax.annotate("", xy=(tx, ty), xytext=(sx, sy), arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.6))
        ax.scatter([sx], [sy], s=330, color="#fff", edgecolors="#111", linewidths=1.6, zorder=5)
        ax.text(sx, sy, str(it["id"]), ha="center", va="center", fontsize=10, fontweight="bold", zorder=6)
    ax.scatter([grid.spawn["x"]], [grid.spawn["y"]], marker="*", s=220, color="#185fa5", zorder=4)
    ax.set_xlabel("metres"); ax.set_ylabel("metres")
    ax.set_title(title, loc="left", fontsize=11)
    handles = [plt.Line2D([], [], marker="s", ls="", color="#1d9e75", label="well observed"),
               plt.Line2D([], [], marker="s", ls="", color="#e24b4a", label="weak (few views / no parallax)"),
               plt.Line2D([], [], marker="s", ls="", color="#7f77dd", label="open frontier"),
               plt.Line2D([], [], color="#185fa5", label="capture path"),
               plt.Line2D([], [], marker="*", ls="", color="#185fa5", label="start"),
               plt.Line2D([], [], marker="o", ls="", color="#fff", markeredgecolor="#111", label="stand here, face along arrow")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=9)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_html(result: dict, out_dir: str) -> str:
    s = result["summary"]
    rows = "".join(
        f"<tr><td><b>{it['id']}</b></td><td>{dict(weak_surface='Weak surface', frontier='Open frontier', ceiling='Ceiling')[it['kind']]}</td>"
        f"<td>{html.escape(it['where'])}<br><span class='k'>map position ({it['stand'][0]:.1f}, {it['stand'][1]:.1f}) m, "
        f"face {it['face_deg']:.0f}&deg;</span></td><td>{html.escape(it['instruction'])}</td></tr>"
        for it in result["items"])
    body = (f"<table><tr><th>#</th><th>Type</th><th>Stand</th><th>What to do</th></tr>{rows}</table>"
            if result["items"] else "<p>Nothing to reshoot.</p>")
    page = f"""<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Recapture ticket</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}}
h1{{font-size:22px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px 10px;border-bottom:1px solid #ddd;
text-align:left;vertical-align:top}}img{{max-width:100%}}.k{{color:#666;font-size:13px}}
.READY{{color:#0f6e56}}.RECAPTURE{{color:#a32d2d}}</style>
<h1>Recapture ticket: <span class='{s['verdict']}'>{s['verdict']}</span></h1>
<p>Well-observed surface: walls and furniture {s['by_band']['mid_pct']:.0f}%, floor
{s['by_band']['floor_pct']:.0f}%, ceiling and upper walls {s['by_band']['high_pct']:.0f}%. Boundary closure
{s['boundary_closure_pct']:.0f}%. {s['blocking_items']} item(s) block training; the rest are optional. {len(result['items'])} item(s), about {s['estimated_recapture_s']} seconds of extra footage.</p>
<img alt='coverage map' src='recapture_ticket.png'>
{body}
<p class='k'>Shoot the items as one or more short clips, then run the linter with --append and re-run the draft.
Start each clip from somewhere already captured so the new footage overlaps the old.</p>"""
    path = os.path.join(out_dir, "recapture_ticket.html")
    with open(path, "w") as f:
        f.write(page)
    return path


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def run(aligned_path: str, out_dir: str, lint_json: str | None = None, cfg: AuditConfig | None = None,
        nav_cfg: ng.NavConfig | None = None) -> dict:
    cfg, nav_cfg = cfg or AuditConfig(), nav_cfg or ng.NavConfig()
    os.makedirs(out_dir, exist_ok=True)
    model = cio.read_model(aligned_path)
    lint = None
    if lint_json and os.path.isfile(lint_json):
        with open(lint_json) as f:
            lint = json.load(f)
    ims = model.sorted_images()
    centers = np.array([im.center() for im in ims])
    fwd0 = ims[0].R()[2, :2]  # world direction of the first camera's optical axis, on the floor plane
    grid = ng.build(model.xyz, centers, fwd0 / max(np.linalg.norm(fwd0), 1e-9), nav_cfg)
    fronts, closure = ng.frontiers(grid, nav_cfg)
    print(f"[audit] {len(ims)} cameras; grid {grid.shape[1]}x{grid.shape[0]} cells; "
          f"floor {grid.stats['floor_area_m2']} m2; {len(fronts)} open frontier(s)")
    obs = observe(model, cfg)
    z = obs["pts"][:, 2]

    def pct(mask):
        return round(100.0 * float((~obs["weak"][mask]).mean()), 1) if mask.any() else 100.0

    items = build_ticket(obs, grid, fronts, model, lint, cfg)
    surface = pct(np.ones(len(z), bool))
    # "ready" is about what a visitor walking the twin will look at: floor, walls, furniture, and a closed
    # boundary. The ceiling is listed on the ticket but does not hold up training.
    walk_view = pct(z <= cfg.ceiling_from)
    blocking = [it for it in items if (it["kind"] == "weak_surface" and it["area_m2"] >= cfg.blocking_m2)
                or it["kind"] == "frontier"]
    ready = (walk_view >= cfg.ready_surface_pct and closure["closure_pct"] >= cfg.ready_closure_pct
             and not blocking)
    summary = dict(
        verdict="READY" if ready else "RECAPTURE",
        surface_quality_pct=surface, walk_view_quality_pct=walk_view, blocking_items=len(blocking),
        by_band=dict(floor_pct=pct(z < 0.1), mid_pct=pct((z >= 0.1) & (z <= cfg.ceiling_from)),
                     high_pct=pct(z > cfg.ceiling_from)),
        boundary_closure_pct=closure["closure_pct"], open_boundary_m=closure["open_m"],
        samples=int(len(z)), median_views=float(np.median(obs["n"])),
        median_baseline_deg=round(float(np.median(obs["baseline"])), 1),
        few_views_pct=round(100.0 * float(obs["few"].mean()), 1),
        narrow_baseline_pct=round(100.0 * float(obs["narrow"].mean()), 1),
        items=len(items), estimated_recapture_s=15 * len(items),
    )
    result = dict(summary=summary, items=items, nav=grid.stats, config=asdict(cfg))
    ov = coverage_overlay(obs, grid, cfg)
    cv2.imwrite(os.path.join(out_dir, "coverage.png"), cv2.cvtColor(ov, cv2.COLOR_RGBA2BGRA))
    ng.save(grid, out_dir, extra=dict(coverage="coverage.png"))
    render_ticket(obs, grid, items, fronts, os.path.join(out_dir, "recapture_ticket.png"),
                  f"{summary['verdict']}  |  walls, furniture, floor {walk_view:.0f}% well observed  |  boundary closure "
                  f"{closure['closure_pct']:.0f}%  |  {len(items)} item(s)", cfg)
    with open(os.path.join(out_dir, "coverage.json"), "w") as f:
        json.dump(result, f, indent=1)
    page = render_html(result, out_dir)
    print(f"[audit] {summary['verdict']}: walk-view surface {walk_view}% (with ceiling {surface}%), "
          f"boundary closure {closure['closure_pct']}%, "
          f"{len(items)} recapture item(s)")
    print(f"[audit] ticket: {page}")
    return result
