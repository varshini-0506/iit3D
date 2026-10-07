"""Metric scale from one printed ArUco marker.

The SOP asks for a single printed marker of known size left on the floor. It is
found automatically in the keyframes, its four corners are triangulated with
the reconstructed camera poses, and the reconstructed edge length is compared
with the printed one. That gives either the scale itself (for unscaled SfM) or
an independent check of a metric model's scale, with no tape measure and no
manual clicking.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from . import colmap_io as cio


def make_marker_png(path: str, marker_id: int = 0, dictionary: str = "DICT_4X4_50", pixels: int = 1200,
                    border: float = 0.15) -> None:
    """Write a printable marker with a white quiet zone."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    img = cv2.aruco.generateImageMarker(d, marker_id, pixels)
    pad = int(pixels * border)
    cv2.imwrite(path, cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255))


def detect(gray: np.ndarray, dictionary: str = "DICT_4X4_50", refine: str = "APRILTAG") -> dict[int, np.ndarray]:
    """Return {marker_id: 4x2 corner pixels} for one image.

    AprilTag-style edge refinement is the default because, measured against
    ground truth on the synthetic room, it is unbiased, whereas sub-pixel corner
    refinement pulls corners inward and shrinks a small marker by 1 to 3%.
    """
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_" + refine)
    corners, ids, _ = cv2.aruco.ArucoDetector(d, params).detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def _triangulate(Ps: list[np.ndarray], uvs: list[np.ndarray]) -> np.ndarray:
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(A))
    X = vt[-1]
    return X[:3] / X[3]


def _reproj_err(P: np.ndarray, X: np.ndarray, uv: np.ndarray) -> float:
    x = P @ np.append(X, 1.0)
    return float(np.linalg.norm(x[:2] / x[2] - uv)) if x[2] > 0 else 1e9


def measure(model: cio.Model, images_dir: str, dictionary: str = "DICT_4X4_50", marker_id: int | None = None) -> dict:
    """Triangulate the marker and return its edge length in the model's units."""
    try:
        out = _measure(model, images_dir, dictionary, marker_id, "APRILTAG")
    except ValueError:
        # fewer detections survive the stricter refinement on small or distant markers
        out = _measure(model, images_dir, dictionary, marker_id, "SUBPIX")
    return out


def _measure(model: cio.Model, images_dir: str, dictionary: str, marker_id: int | None, refine: str) -> dict:
    sightings: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for im in model.sorted_images():
        path = os.path.join(images_dir, im.name)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        cam = model.cameras[im.camera_id]
        H, W = gray.shape
        found = detect(gray, dictionary, refine)
        if not found:
            continue
        # map pixels of this (possibly larger) image into the model's (shrunken, centre-cropped) camera
        k = max(cam.width / W, cam.height / H)
        off = np.array([(W * k - cam.width) / 2.0, (H * k - cam.height) / 2.0])
        K = cam.K()
        P = K @ np.hstack([im.R(), im.tvec.reshape(3, 1)])
        for mid, c in found.items():
            px = c * k - off
            if np.any(cam.dist_coeffs()):
                px = cv2.undistortPoints(px.reshape(-1, 1, 2), K, cam.dist_coeffs(), P=K).reshape(4, 2)
            sightings.setdefault(mid, []).append((P, px))
    if not sightings:
        raise ValueError("Marker not found in any keyframe. Print it larger, flatten it, and film it from about 1 m.")
    if marker_id is None:
        marker_id = max(sightings, key=lambda m: len(sightings[m]))
    obs = sightings.get(marker_id, [])
    if len(obs) < 3:
        raise ValueError(f"Marker {marker_id} was seen in only {len(obs)} keyframe(s); need 3 or more from different angles.")

    corners = []
    for ci in range(4):
        Ps = [o[0] for o in obs]
        uvs = [o[1][ci] for o in obs]
        X = _triangulate(Ps, uvs)
        errs = np.array([_reproj_err(P, X, uv) for P, uv in zip(Ps, uvs)])
        good = errs < max(3.0, 2.5 * np.median(errs))
        if good.sum() >= 3 and not good.all():
            X = _triangulate([p for p, g in zip(Ps, good) if g], [u for u, g in zip(uvs, good) if g])
        corners.append(X)
    corners = np.array(corners)
    edges = np.array([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)])
    diags = np.array([np.linalg.norm(corners[0] - corners[2]), np.linalg.norm(corners[1] - corners[3])]) / np.sqrt(2)
    allm = np.concatenate([edges, diags])
    edge = float(allm.mean())
    n = np.cross(corners[1] - corners[0], corners[3] - corners[0])
    return dict(marker_id=int(marker_id), observations=len(obs), refinement=refine.lower(), edge_recon_units=edge,
                spread_pct=float(100 * np.abs(allm - edge).max() / edge),
                corners=corners, normal=n / np.linalg.norm(n))


def estimate(model: cio.Model, images_dir: str, size_m: float, dictionary: str = "DICT_4X4_50",
             marker_id: int | None = None) -> dict:
    """Scale factor (metres per model unit) from a marker with printed edge `size_m`."""
    m = measure(model, images_dir, dictionary, marker_id)
    m["scale"] = size_m / m["edge_recon_units"]
    return m


def check(model_metric: cio.Model, images_dir: str, size_m: float, dictionary: str = "DICT_4X4_50",
          marker_id: int | None = None) -> dict:
    """Scale error of an already-metric model, in percent and in cm per metre."""
    m = measure(model_metric, images_dir, dictionary, marker_id)
    err = (m["edge_recon_units"] - size_m) / size_m
    return dict(marker_id=m["marker_id"], observations=m["observations"], printed_edge_m=size_m,
                measured_edge_m=round(m["edge_recon_units"], 5), scale_error_pct=round(100 * err, 2),
                error_cm_per_m=round(100 * abs(err), 2), spread_pct=round(m["spread_pct"], 2),
                tilt_from_floor_deg=round(float(np.degrees(np.arccos(min(1.0, abs(m["normal"][2]))))), 2))
