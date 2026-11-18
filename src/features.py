"""The live candidate feature table follows the same scaling and rounding as P1."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ._p1_data import SegmentIndex, candidate_rows
from .geo import Geometry, Trip

_CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/features.yaml").read_text(encoding="utf-8-sig"))
FEATURE_COLS = _CONFIG["FEATURE_COLS"]
RADIUS_M = _CONFIG["candidate_radius_m"]
MAX_CANDIDATES = _CONFIG["max_candidates"]


def build_candidates(trip_dir, *, geometry=None, index=None):
    """Return one row per observation and candidate using only inference inputs."""
    if geometry is None:
        geometry = Geometry(Trip.load(Path(trip_dir)))
    rows = []
    for i, ids, distances, fractions, directions, connectivity in candidate_rows(
            geometry, RADIUS_M, MAX_CANDIDATES, index=index):
        speed, dt = geometry.speed[i], geometry.dt[i]
        time = geometry.trip.track[i, 2]
        key = (geometry.trip.trip_id, i, time)
        rows.extend((*key, int(a), round(d, 3), round(f, 6), round(c, 6),
                     round(speed, 3) if np.isfinite(speed) else np.nan,
                     round(dt, 3) if np.isfinite(dt) else np.nan,
                     round(geometry.length[a], 3), rank, int(con))
                    for rank,(a,d,f,c,con) in enumerate(
                        zip(ids, distances, fractions, directions, connectivity), 1))
    return pd.DataFrame(rows, columns=["trip_id","obs_idx","t","arc_id",*FEATURE_COLS])
