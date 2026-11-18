"""Trip input, metric geometry, and label-independent candidate features."""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray

    @classmethod
    def load(cls, folder: Path):
        stem = folder / folder.name
        track = np.loadtxt(stem.with_suffix(".track"), ndmin=2)
        nodes = np.loadtxt(stem.with_suffix(".nodes"), ndmin=2)
        arcs = np.loadtxt(stem.with_suffix(".arcs"), dtype=np.int32, ndmin=2)
        if (track.shape[1], nodes.shape[1], arcs.shape[1]) != (3, 2, 2):
            raise ValueError(f"{stem}: unexpected number of columns")
        if not len(track) or not len(nodes) or not len(arcs):
            raise ValueError(f"{stem}: empty trip input")
        if not np.isfinite(nodes).all() or arcs.min() < 0 or arcs.max() >= len(nodes):
            raise ValueError(f"{stem}: invalid road network")
        return cls(folder.name, track, nodes, arcs)


def load_reference(folder: Path, arc_count):
    route = np.loadtxt(folder / f"{folder.name}.route", dtype=np.int32, ndmin=2)
    if route.shape[1] != 1 or not len(route) or route.min() < 0 or route.max() >= arc_count:
        raise ValueError(f"{folder}: invalid reference route")
    return route[:, 0]


def valid_observations(track):
    return np.isfinite(track).all(axis=1) & (abs(track[:, 0]) <= 180) & (abs(track[:, 1]) <= 90)


def project(lon_lat, origin):
    """Local equirectangular projection using the median GPS latitude."""
    radians = np.deg2rad(lon_lat - origin)
    radians[:, 0] = (radians[:, 0] + np.pi) % (2 * np.pi) - np.pi
    radians[:, 0] *= np.cos(np.deg2rad(origin[1]))
    return 6_371_008.8 * radians


class Geometry:
    def __init__(self, trip):
        self.trip = trip
        self.valid = valid_observations(trip.track)
        if not np.any(self.valid):
            raise ValueError(f"{trip.trip_id}: no valid GPS observations")
        origin = np.median(trip.track[self.valid, :2], axis=0)
        self.gps = project(trip.track[:, :2], origin)
        self.nodes = project(trip.nodes, origin)
        self.start = self.nodes[trip.arcs[:, 0]]
        self.end = self.nodes[trip.arcs[:, 1]]
        self.vector = self.end - self.start
        self.length = np.linalg.norm(self.vector, axis=1)
        self.dt = np.diff(trip.track[:, 2], prepend=np.nan)
        delta = np.diff(self.gps, axis=0, prepend=np.full((1, 2), np.nan))
        displacement = np.linalg.norm(delta, axis=1)
        self.speed = np.divide(displacement, self.dt, out=np.full(len(delta), np.nan), where=self.dt > 0)
        self.heading = np.divide(delta, displacement[:, None], out=np.zeros_like(delta),
                                 where=(displacement[:, None] >= 3) & (self.dt[:, None] > 0) &
                                       (self.dt[:, None] <= 30) & (self.speed[:, None] >= 1) &
                                       (self.speed[:, None] <= 55))

    def measure(self, observation, ids):
        delta = self.gps[observation] - self.start[ids]
        vector = self.vector[ids]
        length = self.length[ids]
        fraction = np.clip(np.einsum("ij,ij->i", delta, vector) / np.maximum(length**2, 1e-12), 0, 1)
        distance = np.linalg.norm(delta - fraction[:, None] * vector, axis=1)
        direction = (vector @ self.heading[observation]) / np.maximum(length, 1e-6)
        return distance, fraction, np.clip(direction, -1, 1)


class SegmentIndex:
    """Grid of arc bounding boxes, followed by exact point-to-segment distances."""
    def __init__(self, geometry, cell_size):
        self.geometry = geometry
        self.cell_size = cell_size
        self.cells = defaultdict(list)
        lower = np.floor(np.minimum(geometry.start, geometry.end) / cell_size).astype(int)
        upper = np.floor(np.maximum(geometry.start, geometry.end) / cell_size).astype(int)
        for arc_id, ((x0, y0), (x1, y1)) in enumerate(zip(lower, upper)):
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    self.cells[x, y].append(arc_id)

    def query(self, obs_idx, radius, limit):
        point = self.geometry.gps[obs_idx]
        x0, y0 = np.floor((point - radius) / self.cell_size).astype(int)
        x1, y1 = np.floor((point + radius) / self.cell_size).astype(int)
        ids = set()
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                ids.update(self.cells.get((x, y), ()))
        ids = np.fromiter(ids, dtype=np.int32)
        distance, fraction, direction = self.geometry.measure(obs_idx, ids)
        keep = np.flatnonzero(distance <= radius)
        order = keep[np.lexsort((ids[keep], -direction[keep], distance[keep]))[:limit]]
        return ids[order], distance[order], fraction[order], direction[order]


def candidate_rows(geometry, radius_m, max_candidates, search_radius_m=None):
    """Yield one batch per valid GPS observation, without reference information."""
    search_radius_m = radius_m if search_radius_m is None else search_radius_m
    if not 0 < radius_m <= search_radius_m or max_candidates <= 0:
        raise ValueError("Candidate radii and limit must be positive and ordered")
    index = SegmentIndex(geometry, search_radius_m)
    previous = None
    arcs = geometry.trip.arcs
    for obs_idx in range(len(geometry.gps)):
        if not geometry.valid[obs_idx]:
            previous = None
            continue
        if not 0 < geometry.dt[obs_idx] <= 30:
            previous = None
        ids, distance, fraction, direction = index.query(obs_idx, search_radius_m, max_candidates)
        keep = distance <= radius_m
        ids, distance, fraction, direction = (a[keep] for a in (ids, distance, fraction, direction))
        connects = np.zeros(len(ids), dtype=bool)
        if previous is not None:
            connects = (ids == previous) | (arcs[previous, 1] == arcs[ids, 0])
        yield obs_idx, ids, distance, fraction, direction, connects
        previous = ids[0] if len(ids) else None
