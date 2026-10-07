"""Capture linter: grade a walkthrough video and pick the keyframes worth reconstructing.

Reconstruction quality is decided at capture time, so this stage runs first and
takes seconds, not hours. Each analysed frame gets four measurements:

    sharpness   fine-detail to coarse-detail energy ratio, judged against neighbouring frames
    motion      median sparse optical flow, in image-widths per second
    texture     how many trackable corners the frame has and how much of it they cover
    exposure    clipped highlights / crushed shadows / auto-exposure hunting

Keyframes are then chosen by *parallax*, not by a fixed frame rate: a new
keyframe is taken once the scene has moved a set fraction of the image width,
and within that window the sharpest frame wins. Standing still adds no frames,
fast pans add more, and motion-blurred frames are skipped.
"""

from __future__ import annotations

import base64
import html
import json
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class LintConfig:
    analysis_width: int = 480      # frames are analysed at this width
    analysis_fps: float = 15.0     # analyse at most this many frames per second
    keyframe_step: float = 0.07    # parallax between keyframes, as a fraction of image width
    max_keyframes: int = 400       # step is widened automatically to stay under this
    max_side: int = 1600           # longest side of saved keyframes, pixels
    jpeg_quality: int = 95
    blur_rel: float = 0.45         # blurry if sharpness < this x local reference
    blur_window_s: float = 1.5     # half-width of the local sharpness reference window
    fast_warn: float = 0.5         # widths/second; ~30 deg/s on a phone main camera
    fast_fail: float = 1.0         # widths/second; motion blur is near certain
    min_features: int = 40         # fewer trackable corners than this = texture-poor
    min_cover: float = 0.35        # ...or corners present in less than this share of the frame
    clip_frac: float = 0.25        # fraction of blown/crushed pixels that counts as bad exposure
    hunt_rate: float = 0.35        # relative brightness change per second while nearly still
    min_keyframes: int = 30        # fewer than this cannot make a usable twin


# --------------------------------------------------------------------------
# pass 1: per-frame measurements
# --------------------------------------------------------------------------

def _open(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # honour phone rotation metadata
    return cap


def _sharpness(gray: np.ndarray, edge: float = 12.0) -> float:
    """Blur measure that does not depend on how much texture is in view.

    Variance of the Laplacian drops on a blank wall just as it does under blur,
    so it cannot tell the two apart. Instead, look only at pixels on or next to
    an edge and compare second-derivative energy with first-derivative energy,
    per axis: blur spreads an edge out, so the ratio falls, whether the frame
    holds one edge or a thousand. The minimum over x and y catches directional
    motion blur. Returns NaN when the frame has no edges to judge by.
    """
    g = gray.astype(np.float32)
    k = np.ones((1, 5), np.uint8)
    out = []
    for axis in (1, 0):
        a = g if axis == 1 else g.T
        d1 = a[:, 2:] - a[:, :-2]
        d2 = a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]
        mask = cv2.dilate((np.abs(d1) > edge).astype(np.uint8), k) > 0
        if mask.mean() < 0.002:
            continue
        out.append(float((d2[mask] ** 2).sum() / max(float((d1[mask] ** 2).sum()), 1e-6)))
    return min(out) if out else float("nan")


def _grid_cover(pts, width: int, height: int, nx: int = 8, ny: int = 6) -> float:
    """Share of an nx-by-ny grid over the frame that holds at least one trackable corner."""
    if pts is None or len(pts) == 0:
        return 0.0
    p = pts.reshape(-1, 2)
    cx = np.clip((p[:, 0] * nx / width).astype(int), 0, nx - 1)
    cy = np.clip((p[:, 1] * ny / height).astype(int), 0, ny - 1)
    return len(np.unique(cy * nx + cx)) / float(nx * ny)


def analyse_video(path: str, cfg: LintConfig) -> dict:
    """Measure every analysed frame. Returns arrays keyed by measurement name."""
    cap = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    stride = max(1, int(round(fps / cfg.analysis_fps)))
    idx, t, sharp, disp, feats, cover, luma, clip_hi, clip_lo = [], [], [], [], [], [], [], [], []
    prev, prev_pts, width, height, i = None, None, 0, 0, -1
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    while True:
        if not cap.grab():
            break
        i += 1
        if i % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue
        height, width = frame.shape[:2]
        aw = cfg.analysis_width
        ah = max(8, int(round(height * aw / width)))
        gray = cv2.cvtColor(cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

        sharp.append(_sharpness(gray))
        luma.append(float(gray.mean()))
        clip_hi.append(float((gray > 250).mean()))
        clip_lo.append(float((gray < 8).mean()))

        d, nfeat = 0.0, 0
        if prev is not None and prev_pts is not None and len(prev_pts) >= 8:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, prev_pts, None, **lk)
            good = st.ravel() == 1
            nfeat = int(good.sum())
            if nfeat >= 8:
                mag = np.linalg.norm((nxt - prev_pts).reshape(-1, 2)[good], axis=1)
                d = float(np.median(mag)) / aw
            else:
                d = float("nan")  # tracking lost between these two frames
        elif prev is not None:
            d = float("nan")
        disp.append(d)
        feats.append(nfeat)
        idx.append(i)
        t.append(i / fps)

        prev = gray
        prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=8)
        cover.append(_grid_cover(prev_pts, aw, ah))
    cap.release()
    if len(idx) < 2:
        raise ValueError(f"{path}: fewer than two readable frames")

    feats = np.array(feats, float)
    feats[0] = feats[1]  # first frame has no predecessor
    return dict(path=path, fps=float(fps), stride=stride, width=width, height=height,
                frame=np.array(idx), t=np.array(t), sharp=np.array(sharp), disp=np.array(disp),
                feats=feats, cover=np.array(cover), luma=np.array(luma), clip_hi=np.array(clip_hi), clip_lo=np.array(clip_lo))


# --------------------------------------------------------------------------
# flags, segments
# --------------------------------------------------------------------------

def _rolling_percentile(x: np.ndarray, half: int, q: float) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.percentile(x[max(0, i - half): i + half + 1], q)
    return out


def flag_frames(m: dict, cfg: LintConfig) -> dict:
    dt = m["stride"] / m["fps"]
    half = max(2, int(round(cfg.blur_window_s / dt)))
    no_edges = ~np.isfinite(m["sharp"])
    sharp = np.where(no_edges, np.nanmedian(m["sharp"]) if (~no_edges).any() else 1.0, m["sharp"])
    ref = _rolling_percentile(sharp, half, 90)
    rel = sharp / np.maximum(ref, 1e-9)
    lost = ~np.isfinite(m["disp"])
    speed = np.where(lost, 0.0, m["disp"]) / dt  # widths per second
    low_tex = (m["feats"] < cfg.min_features) | (m["cover"] < cfg.min_cover) | lost | no_edges
    low_tex[0] = low_tex[1]
    blurry = (rel < cfg.blur_rel) & ~low_tex  # with little to look at, softness cannot be judged
    fast = speed > cfg.fast_warn
    very_fast = speed > cfg.fast_fail
    bad_expo = (m["clip_hi"] > cfg.clip_frac) | (m["clip_lo"] > cfg.clip_frac)
    dl = np.abs(np.diff(m["luma"], prepend=m["luma"][0])) / np.maximum(m["luma"], 1.0) / dt
    hunting = (dl > cfg.hunt_rate) & (speed < 0.15)
    return dict(rel_sharp=rel, sharp=sharp, speed=speed, blurry=blurry, fast=fast, very_fast=very_fast,
                low_texture=low_tex, bad_exposure=bad_expo, hunting=hunting)


def _segments(flag: np.ndarray, t: np.ndarray, dt: float, min_len: float = 0.3, gap: float = 0.4) -> list[tuple[float, float]]:
    segs, start, last = [], None, None
    for i, f in enumerate(flag):
        if f:
            if start is None:
                start = t[i]
            elif t[i] - last > gap + dt:
                segs.append((start, last + dt))
                start = t[i]
            last = t[i]
    if start is not None:
        segs.append((start, last + dt))
    merged: list[tuple[float, float]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(float(s), float(e)) for s, e in merged if e - s >= min_len]


ADVICE = {
    "blurry": "Frames here are soft. Slow down, hold the phone with both hands, add light.",
    "too_fast": "Camera moved too fast. Reshoot this stretch at half the speed; turn with your feet, not your wrist.",
    "low_texture": "Almost nothing to track (blank wall, glass or darkness). Step back so furniture or edges share the frame.",
    "bad_exposure": "Large blown-out or pitch-black areas. Close blinds on bright windows, switch lights on, lock exposure.",
    "exposure_hunting": "Brightness pumped while the camera was nearly still. Lock exposure and white balance before recording.",
}


def build_issues(m: dict, fl: dict, cfg: LintConfig) -> list[dict]:
    dt = m["stride"] / m["fps"]
    issues = []
    for key, flag, sev in (("blurry", fl["blurry"], "warn"),
                           ("too_fast", fl["fast"], "warn"),
                           ("low_texture", fl["low_texture"], "warn"),
                           ("bad_exposure", fl["bad_exposure"], "warn"),
                           ("exposure_hunting", fl["hunting"], "info")):
        for s, e in _segments(flag, m["t"], dt):
            severity = sev
            if key == "too_fast":
                inside = (m["t"] >= s) & (m["t"] <= e)
                if fl["very_fast"][inside].any():
                    severity = "fail"
            issues.append(dict(type=key, t_start=round(s, 2), t_end=round(e, 2), severity=severity, advice=ADVICE[key]))
    issues.sort(key=lambda d: d["t_start"])
    return issues


# --------------------------------------------------------------------------
# keyframe selection by parallax
# --------------------------------------------------------------------------

def select_keyframes(m: dict, fl: dict, cfg: LintConfig) -> tuple[list[int], list[int], float]:
    """Return (selected analysed-frame indices, weak-link indices, step used)."""
    d = np.where(np.isfinite(m["disp"]), m["disp"], cfg.keyframe_step)  # a tracking break forces a keyframe
    cum = np.cumsum(d)
    step = max(cfg.keyframe_step, cum[-1] / max(1, cfg.max_keyframes))
    usable = ~fl["blurry"] & ~fl["very_fast"]
    sharp = fl["sharp"]
    n = len(cum)
    sel, weak = [], []
    # start on the sharpest frame of the first half-step
    first = np.flatnonzero(cum <= 0.5 * step)
    first = first if len(first) else np.array([0])
    cur = int(first[np.argmax(np.where(usable[first], sharp[first], -1))])
    sel.append(cur)
    while True:
        rel = cum - cum[cur]
        cand = np.flatnonzero((rel >= 0.7 * step) & (rel <= 1.3 * step) & (np.arange(n) > cur))
        if len(cand) == 0:
            nxt = np.flatnonzero((rel >= 0.7 * step) & (np.arange(n) > cur))
            if len(nxt) == 0:
                break
            cand = nxt[:1]
        good = cand[usable[cand]]
        if len(good):
            cur = int(good[np.argmax(sharp[good])])
        else:  # nothing clean in the window: keep the chain connected, but record it
            cur = int(cand[np.argmax(sharp[cand])])
            weak.append(cur)
        sel.append(cur)
    return sel, weak, float(step)


def extract_keyframes(path: str, frame_ids: list[int], names: list[str], out_dir: str, cfg: LintConfig) -> tuple[int, int]:
    os.makedirs(out_dir, exist_ok=True)
    want = dict(zip(frame_ids, names))
    last = max(frame_ids)
    cap = _open(path)
    i, size = -1, (0, 0)
    while i < last:
        if not cap.grab():
            break
        i += 1
        if i not in want:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        s = cfg.max_side / max(h, w)
        if s < 1:
            frame = cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        size = (frame.shape[1], frame.shape[0])
        cv2.imwrite(os.path.join(out_dir, want[i]), frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    cap.release()
    return size


# --------------------------------------------------------------------------
# loop closure: does the walk end where it began?
# --------------------------------------------------------------------------

def check_loop_closure(kf_paths: list[str], k: int = 4) -> dict:
    if len(kf_paths) < 3 * k:
        return dict(closed=None, inliers=0)
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def feats(p):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        s = 640 / max(img.shape)
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return orb.detectAndCompute(img, None)

    heads = [feats(p) for p in kf_paths[:k]]
    tails = [feats(p) for p in kf_paths[-k:]]
    best = 0
    for ka, da in heads:
        for kb, db in tails:
            if da is None or db is None or len(da) < 20 or len(db) < 20:
                continue
            pairs = [p for p in bf.knnMatch(da, db, k=2) if len(p) == 2]
            good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
            if len(good) < 15:
                continue
            pa = np.float32([ka[g.queryIdx].pt for g in good])
            pb = np.float32([kb[g.trainIdx].pt for g in good])
            _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.99)
            if mask is not None:
                best = max(best, int(mask.sum()))
    return dict(closed=bool(best >= 25), inliers=best)


# --------------------------------------------------------------------------
# one clip, end to end
# --------------------------------------------------------------------------

def lint_clip(path: str, out_dir: str, clip_index: int, cfg: LintConfig) -> dict:
    m = analyse_video(path, cfg)
    fl = flag_frames(m, cfg)
    sel, weak, step = select_keyframes(m, fl, cfg)
    names = [f"c{clip_index:02d}_{int(m['frame'][i]):06d}.jpg" for i in sel]
    kf_dir = os.path.join(out_dir, "keyframes")
    size = extract_keyframes(path, [int(m["frame"][i]) for i in sel], names, kf_dir, cfg)
    loop = check_loop_closure([os.path.join(kf_dir, n) for n in names])
    issues = build_issues(m, fl, cfg)

    n = len(m["t"])
    frac = {k: float(fl[k].mean()) for k in ("blurry", "fast", "very_fast", "low_texture", "bad_exposure", "hunting")}
    weak_frac = len(weak) / max(1, len(sel))
    score = 100.0 - 100.0 * (0.6 * frac["blurry"] + 0.4 * frac["fast"] + 0.6 * frac["very_fast"]
                             + 0.4 * frac["low_texture"] + 0.3 * frac["bad_exposure"]) - 60.0 * weak_frac
    if loop["closed"] is False:
        score -= 8.0
    score = float(np.clip(score, 0, 100))
    reasons = []
    if len(sel) < cfg.min_keyframes:
        reasons.append(f"only {len(sel)} keyframes (need at least {cfg.min_keyframes}); the clip is too short or too static")
    if weak_frac > 0.2:
        reasons.append(f"{weak_frac:.0%} of keyframes had no clean frame to choose from")
    if frac["very_fast"] > 0.15:
        reasons.append(f"camera moved far too fast for {frac['very_fast']:.0%} of the clip")
    if score < 50:
        reasons.append(f"capture score {score:.0f} is below 50")
    verdict = "FAIL" if reasons else ("WARN" if score < 75 else "PASS")

    duration = float(m["t"][-1] + m["stride"] / m["fps"])
    keyframes = [dict(name=nm, frame=int(m["frame"][i]), t=round(float(m["t"][i]), 3),
                      sharpness=round(float(fl["sharp"][i]), 4), weak=bool(i in weak)) for nm, i in zip(names, sel)]
    ds = max(1, n // 600)  # keep the stored timeline small
    return dict(
        clip=clip_index, source=os.path.basename(path), verdict=verdict, score=round(score, 1), fail_reasons=reasons,
        duration_s=round(duration, 2), fps=round(m["fps"], 3), resolution=[m["width"], m["height"]],
        keyframe_size=list(size), frames_analysed=n, keyframes=len(sel), keyframe_step=round(step, 4),
        weak_links=len(weak), loop_closure=loop,
        fractions={k: round(v, 4) for k, v in frac.items()},
        median_speed=round(float(np.median(fl["speed"])), 3), p95_speed=round(float(np.percentile(fl["speed"], 95)), 3),
        issues=issues, keyframe_list=keyframes,
        timeline=dict(t=[round(float(v), 2) for v in m["t"][::ds]],
                      rel_sharp=[round(float(v), 3) for v in fl["rel_sharp"][::ds]],
                      speed=[round(float(v), 3) for v in fl["speed"][::ds]],
                      feats=[int(v) for v in m["feats"][::ds]],
                      cover=[round(float(v), 2) for v in m["cover"][::ds]],
                      luma=[round(float(v), 1) for v in m["luma"][::ds]]),
    )


# --------------------------------------------------------------------------
# report rendering
# --------------------------------------------------------------------------

_COLORS = {"blurry": "#d85a30", "too_fast": "#ba7517", "low_texture": "#7f77dd",
           "bad_exposure": "#888780", "exposure_hunting": "#b4b2a9"}


def render_timeline(clip: dict, path: str, cfg: LintConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tl = clip["timeline"]
    t = np.array(tl["t"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.6), sharex=True, constrained_layout=True)
    axes[0].plot(t, tl["rel_sharp"], color="#185fa5", lw=1)
    axes[0].axhline(cfg.blur_rel, color="#d85a30", ls="--", lw=0.8)
    axes[0].set_ylabel("sharpness\n(vs neighbours)")
    axes[0].set_ylim(0, 1.3)
    axes[1].plot(t, tl["speed"], color="#0f6e56", lw=1)
    axes[1].axhline(cfg.fast_warn, color="#ba7517", ls="--", lw=0.8)
    axes[1].axhline(cfg.fast_fail, color="#a32d2d", ls="--", lw=0.8)
    axes[1].set_ylabel("speed\n(widths / s)")
    axes[1].set_ylim(0, max(1.2, float(np.percentile(tl["speed"], 99)) * 1.1))
    axes[2].plot(t, tl["cover"], color="#534ab7", lw=1)
    axes[2].axhline(cfg.min_cover, color="#7f77dd", ls="--", lw=0.8)
    axes[2].set_ylabel("frame with\ntrackable detail")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlabel("time in clip (s)")
    for ax in axes:
        for iss in clip["issues"]:
            ax.axvspan(iss["t_start"], iss["t_end"], color=_COLORS.get(iss["type"], "#888"), alpha=0.18, lw=0)
        for kf in clip["keyframe_list"]:
            ax.axvline(kf["t"], color="#000", alpha=0.07, lw=0.6)
        ax.grid(alpha=0.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_title(f"{clip['source']}  |  {clip['verdict']}  score {clip['score']:.0f}  |  "
                      f"{clip['keyframes']} keyframes (thin vertical lines)", loc="left", fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _mmss(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def render_html(result: dict, out_dir: str, cfg: LintConfig) -> str:
    parts = ["<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Capture lint report</title>",
             "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}"
             "h1{font-size:22px}h2{font-size:17px;margin-top:28px}table{border-collapse:collapse;width:100%}"
             "td,th{padding:6px 10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}"
             ".PASS{color:#0f6e56}.WARN{color:#854f0b}.FAIL{color:#a32d2d}img{max-width:100%}"
             ".k{color:#666}</style>",
             f"<h1>Capture lint report: <span class='{result['verdict']}'>{result['verdict']}</span></h1>",
             f"<p>{result['total_keyframes']} keyframes from {len(result['clips'])} clip(s). "
             f"Overall capture score {result['score']:.0f} / 100.</p>"]
    for clip in result["clips"]:
        png = os.path.join(out_dir, f"lint_timeline_c{clip['clip']:02d}.png")
        render_timeline(clip, png, cfg)
        with open(png, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        fr = clip["fractions"]
        lc = clip["loop_closure"]["closed"]
        parts += [f"<h2>Clip {clip['clip']}: {html.escape(clip['source'])} "
                  f"<span class='{clip['verdict']}'>{clip['verdict']}</span></h2>",
                  f"<p class='k'>{clip['duration_s']:.0f} s, {clip['resolution'][0]}x{clip['resolution'][1]} at "
                  f"{clip['fps']:.0f} fps. Score {clip['score']:.0f}. {clip['keyframes']} keyframes, "
                  f"{clip['weak_links']} weak link(s). Blurry {fr['blurry']:.0%}, too fast {fr['fast']:.0%}, "
                  f"texture-poor {fr['low_texture']:.0%}. Loop closed: "
                  f"{'yes' if lc else 'no' if lc is False else 'n/a'}.</p>"]
        for r in clip["fail_reasons"]:
            parts.append(f"<p class='FAIL'>Fails because: {html.escape(r)}.</p>")
        parts.append(f"<img alt='timeline' src='data:image/png;base64,{b64}'>")
        if clip["issues"]:
            parts.append("<table><tr><th>When</th><th>Problem</th><th>What to do</th></tr>")
            for iss in clip["issues"]:
                parts.append(f"<tr><td>{_mmss(iss['t_start'])} to {_mmss(iss['t_end'])}</td>"
                             f"<td>{iss['type'].replace('_', ' ')} ({iss['severity']})</td>"
                             f"<td>{html.escape(iss['advice'])}</td></tr>")
            parts.append("</table>")
        else:
            parts.append("<p>No problem segments found.</p>")
    path = os.path.join(out_dir, "lint_report.html")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    return path


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def run(videos: list[str], out_dir: str, cfg: LintConfig | None = None, append: bool = False) -> dict:
    cfg = cfg or LintConfig()
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "lint.json")
    clips: list[dict] = []
    if append and os.path.isfile(json_path):
        with open(json_path) as f:
            clips = json.load(f)["clips"]
    elif not append:
        kf_dir = os.path.join(out_dir, "keyframes")
        if os.path.isdir(kf_dir):
            for n in os.listdir(kf_dir):
                if n.lower().endswith(".jpg"):
                    os.remove(os.path.join(kf_dir, n))
    next_idx = max([c["clip"] for c in clips], default=-1) + 1
    for k, v in enumerate(videos):
        print(f"[lint] analysing {v}")
        clip = lint_clip(v, out_dir, next_idx + k, cfg)
        print(f"[lint]   {clip['verdict']}  score {clip['score']:.0f}  keyframes {clip['keyframes']}  "
              f"issues {len(clip['issues'])}")
        clips.append(clip)

    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    # recapture clips are short top-ups: only the first clip must stand on its own
    main = clips[0]
    verdict = main["verdict"]
    for c in clips[1:]:
        if c["verdict"] == "FAIL" and any("too fast" in r or "no clean frame" in r for r in c["fail_reasons"]):
            verdict = max(verdict, "WARN", key=order.get)
    total_dur = sum(c["duration_s"] for c in clips)
    result = dict(
        verdict=verdict,
        score=round(sum(c["score"] * c["duration_s"] for c in clips) / max(total_dur, 1e-6), 1),
        total_keyframes=sum(c["keyframes"] for c in clips),
        total_duration_s=round(total_dur, 2),
        config=asdict(cfg), clips=clips,
    )
    with open(json_path, "w") as f:
        json.dump(result, f, indent=1)
    # keyframes kept only to hold the chain together: fine for pose estimation, left out of splat training
    with open(os.path.join(out_dir, "weak_keyframes.txt"), "w") as f:
        for c in clips:
            for kf in c["keyframe_list"]:
                if kf["weak"]:
                    f.write(kf["name"] + "\n")
    report = render_html(result, out_dir, cfg)
    print(f"[lint] {result['verdict']}: {result['total_keyframes']} keyframes -> {os.path.join(out_dir, 'keyframes')}")
    print(f"[lint] report: {report}")
    return result
