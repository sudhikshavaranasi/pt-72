"""Directed-road candidate HMM. The inference entry point reads no reference routes."""
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
import json
from pathlib import Path

import numpy as np

from .geo import Geometry, Trip

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/hmm.yaml"


def load_config(path=DEFAULT_CONFIG):
    # JSON is a valid subset of YAML. Also accept team-edited YAML if available.
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(text)


@dataclass
class Scores:
    obs: np.ndarray
    arcs: np.ndarray
    probabilities: np.ndarray

    @classmethod
    def from_frame(cls, frame, trip):
        if len(frame) and not frame["trip_id"].eq(trip.trip_id).all():
            raise ValueError("Candidate rows contain a different trip ID")
        obs = frame["obs_idx"].to_numpy(dtype=np.int64)
        timestamps = frame["t"].to_numpy(dtype=float)
        if np.any((obs < 0) | (obs >= len(trip.track))):
            raise ValueError("Candidate observation index is outside this trip")
        if not np.allclose(timestamps, trip.track[obs, 2], rtol=0, atol=1e-6):
            raise ValueError("Candidate timestamps do not match the track")
        return cls(obs, frame["arc_id"].to_numpy(dtype=np.int64),
                   frame["score"].to_numpy(dtype=float))


class RoadGraph:
    """Directed adjacency, with cached bounded Dijkstra from arc end nodes."""
    def __init__(self, geometry):
        self.adjacency = [[] for _ in geometry.nodes]
        for (start, end), length in zip(geometry.trip.arcs, geometry.length):
            self.adjacency[start].append((int(end), float(length)))
        self.distances = lru_cache(maxsize=4096)(self._distances)

    def _distances(self, source, cutoff):
        tentative = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, node = heappop(queue)
            if distance != tentative[node]:
                continue
            for neighbor, length in self.adjacency[node]:
                new = distance + length
                if new <= cutoff and new < tentative.get(neighbor, np.inf):
                    tentative[neighbor] = new
                    heappush(queue, (new, neighbor))
        return tentative


@dataclass
class Step:
    network_m: np.ndarray
    observed_m: float
    dt: float
    stationary: bool
    speed_spike: bool
    same_arc: np.ndarray


def bag_candidates(scores, geometry):
    n = len(geometry.gps)
    obs = np.asarray(scores.obs)
    arcs = np.asarray(scores.arcs)
    probability = np.asarray(scores.probabilities)
    if not (len(obs) == len(arcs) == len(probability)):
        raise ValueError("Candidate columns differ in length")
    if (np.any((obs < 0) | (obs >= n)) or np.any((arcs < 0) | (arcs >= len(geometry.length)))
            or np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1))):
        raise ValueError("Invalid observation, arc ID or score")
    if np.any(~geometry.valid[obs]):
        raise ValueError("Candidates supplied for invalid GPS observations")
    order = np.argsort(obs, kind="stable")
    starts = np.searchsorted(obs[order], np.arange(n+1))
    states = []
    for i in range(n):
        rows = order[starts[i]:starts[i+1]]
        ids = arcs[rows]
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Candidate arc IDs must be unique per observation")
        weights = probability[rows]
        _, fractions, _ = geometry.measure(i, ids) if len(ids) else (None, np.empty(0), None)
        states.append((ids, weights, fractions))
    return states


def transition_step(geometry, graph, i, previous, current, config):
    prev, _, f0 = previous
    following, _, f1 = current
    dt = float(geometry.dt[i])
    gps = float(np.linalg.norm(geometry.gps[i] - geometry.gps[i-1]))
    spike = gps / dt > config["speed_spike_m_s"]
    observed = min(gps, config["speed_spike_m_s"] * dt)
    stationary = gps <= config["stationary_m"] and dt <= config["stationary_max_dt_s"]
    limit = min(config["max_road_m"], observed + config["road_slack_m"] + config["road_slack_m_s"] * dt)
    bucket = next((x for x in (80, 150, 300, 600, 1200) if x >= limit), int(np.ceil(limit)))
    arcs = geometry.trip.arcs
    from_nodes, to_nodes = arcs[following, 0], arcs[prev, 1]
    middle = np.empty((len(prev), len(following)), dtype=float)
    for j, source in enumerate(to_nodes):
        distances = graph.distances(int(source), bucket)
        middle[j] = [distances.get(int(target), np.inf) for target in from_nodes]
    dist = (geometry.length[prev]*(1-f0))[:, None] + middle + geometry.length[following]*f1
    same = prev[:, None] == following
    direct = geometry.length[prev][:, None] * (f1 - f0[:, None])
    forward = same & (direct >= -config["backtrack_tolerance_m"])
    dist = np.where(forward, np.minimum(dist, np.maximum(0, direct)), dist)
    if stationary:
        dist = np.where(same & (abs(direct) <= config["backtrack_tolerance_m"]), 0, dist)
    dist[dist > limit] = np.inf
    return Step(dist, observed, dt, stationary, spike, same)


def prepare(geometry, scores, config, *, graph=None):
    states = bag_candidates(scores, geometry)
    own_graph = graph is None
    if own_graph:
        graph = RoadGraph(geometry)
    steps = [None] * len(states)
    for i in range(1, len(states)):
        dt = geometry.dt[i]
        if len(states[i-1][0]) and len(states[i][0]) and 0 < dt <= config["gap_threshold_s"]:
            steps[i] = transition_step(geometry, graph, i, states[i-1], states[i], config)
    if own_graph:
        graph.distances.cache_clear()
    return states, steps


def transition_log(step, config):
    beta = config["beta_m"] + config["beta_m_per_s"] * max(0, step.dt - 1)
    if step.speed_spike:
        beta *= config["spike_beta_multiplier"]
    cost = abs(step.network_m - step.observed_m) / beta
    if step.stationary:
        cost = step.network_m / beta + config["stationary_switch_cost"] * (~step.same_arc)
    return -cost


def viterbi(prepared, config):
    states, steps = prepared
    n = len(states)
    clf = np.full(n, -1, dtype=np.int32)
    pred = clf.copy()
    piece = clf.copy()
    parents = [None] * n
    last = None
    active = -1
    objective = None
    splits = {"zero_candidates": 0, "gap_or_clock": 0, "unreachable": 0}

    def finish():
        if last is None:
            return
        pos = int(np.argmax(objective))
        i = last
        while True:
            pred[i] = states[i][0][pos]
            if parents[i] is None:
                break
            pos = int(parents[i][pos])
            i -= 1

    for i, ((ids, raw, _), step) in enumerate(zip(states, steps)):
        if not len(ids):
            splits["zero_candidates"] += 1
            finish()
            last = None
            continue
        clf[i] = ids[raw.argmax()]
        scores = np.clip(raw, 1e-9, 1)
        emission = config["emission_weight"] * np.log(scores / scores.sum())
        start_piece = last is None
        if not start_piece and step is None:
            splits["gap_or_clock"] += 1
            start_piece = True
        if not start_piece:
            values = objective[:, None] + transition_log(step, config)
            parent = values.argmax(axis=0)
            updated = values[parent, np.arange(len(ids))] + emission
            if np.any(np.isfinite(updated)):
                parents[i] = parent
                objective = updated - np.max(updated)
            else:
                splits["unreachable"] += 1
                start_piece = True
        if start_piece:
            finish()
            active += 1
            objective = emission - np.max(emission)
        piece[i] = active
        last = i
    finish()
    return clf, pred, piece, splits


def route_records(trip, pred, piece):
    records = []
    previous = None
    previous_piece = None
    for arc, segment in zip(pred, piece):
        if arc >= 0 and (arc != previous or segment != previous_piece):
            coordinates = trip.nodes[trip.arcs[arc]][:, ::-1].tolist()
            records.append(dict(arc_id=int(arc), coords=coordinates))
        previous, previous_piece = arc, segment
    return records


def decode(trip_dir, cand_df_with_scores, *, config_path=DEFAULT_CONFIG, geometry=None, graph=None):
    """Accept one trip's candidate DataFrame and return the team contract payload."""
    import pandas as pd

    if geometry is None:
        geometry = Geometry(Trip.load(Path(trip_dir)))
    trip = geometry.trip
    config = load_config(config_path)
    scores = Scores.from_frame(cand_df_with_scores, trip)
    clf, hmm, piece, _ = viterbi(prepare(geometry, scores, config, graph=graph), config)
    points = pd.DataFrame(dict(trip_id=trip.trip_id, obs_idx=np.arange(len(trip.track)),
                              t=trip.track[:, 2], pred_arc_clf=clf, pred_arc_hmm=hmm, piece_id=piece))
    return dict(points=points, route=route_records(trip, hmm, piece), hmm_config=config)


