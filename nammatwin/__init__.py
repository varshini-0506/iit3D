"""NammaTwin: closed-loop capture-to-digital-twin pipeline (Round 1).

Stages (each is a subcommand of ``python -m nammatwin``):

    lint      video -> quality checks + sharp, well-spaced keyframes
    align     any SfM model -> gravity-aligned, Manhattan-aligned, metric frame
    audit     aligned draft -> coverage heatmap + recapture ticket
    nav       aligned points / splat -> walkable grid + floor plan
    report    everything -> one report card (report.json) for the viewer
"""

__version__ = "0.1.0"
