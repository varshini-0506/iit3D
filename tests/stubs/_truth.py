"""Ground truth shared by the GPU-stage stand-ins: the synthetic room and where each keyframe really was."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from nammatwin import synth  # noqa: E402

SIZE = (640, 360)
FPS = 10.0


def keyframe_poses(work: str):
    """(room, names, camera-to-world matrices) for every keyframe listed in work/lint.json."""
    room = synth.Room()
    tables = {0: synth.walk_trajectory(room, FPS)[1], 1: synth.recapture_trajectory(room, FPS)[1]}
    with open(os.path.join(work, "lint.json")) as f:
        lint = json.load(f)
    names, c2ws = [], []
    for clip in lint["clips"]:
        for kf in clip["keyframe_list"]:
            names.append(kf["name"])
            c2ws.append(tables[clip["clip"]][kf["frame"]])
    order = sorted(range(len(names)), key=lambda i: names[i])
    return room, [names[i] for i in order], [c2ws[i] for i in order]
