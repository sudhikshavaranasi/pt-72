"""Run P1 on the student dataset and export the P2/P3 handoff."""
import argparse
import csv
from contextlib import ExitStack
import gzip
import json
from pathlib import Path

import networkx as nx
import numpy as np

from alignment import align_reference
from data import Geometry, Trip, candidate_rows, load_reference
from maps import write_map

RADII = (25, 40, 60, 80, 120)
MAX_CANDIDATES = 30
FEATURES = (
    "split", "trip_id", "observation_index", "timestamp_s", "arc_id", "distance_m",
    "projection_fraction", "direction_cosine", "speed_m_s", "time_gap_s",
    "segment_length_m", "distance_rank", "connects_previous_best", "label",
)
REFERENCES = (
    "split", "trip_id", "observation_index", "timestamp_s", "reference_arc_id",
    "reference_route_position", "alignment_distance_m", "alignment_flag", "candidate_count", "candidate_miss",
)
if __name__ != "__main__":
    __all__ = ["FEATURES", "REFERENCES"]


def inventory(folder, geometry, route):
    trip = geometry.trip
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(trip.nodes)))
    graph.add_edges_from((int(a), int(b)) for a, b in trip.arcs)
    components = [len(group) for group in nx.weakly_connected_components(graph)]
    one_way = sum(a != b and not graph.has_edge(b, a) for a, b in graph.edges)
    ordered = trip.arcs[route]
    valid_coord = trip.track[geometry.valid, :2]
    dt = geometry.dt[np.isfinite(geometry.dt)]
    speed = geometry.speed[np.isfinite(geometry.speed)]
    files = {f"{ext}_bytes": (folder / f"{folder.name}.{ext}").stat().st_size
             for ext in ("track", "nodes", "arcs", "route")}
    return dict(
        gps_observations=len(trip.track), invalid_gps=int((~geometry.valid).sum()),
        missing_gps_values=int(np.isnan(trip.track).sum()), nodes=len(trip.nodes),
        directed_arcs=len(trip.arcs), distinct_directed_pairs=graph.number_of_edges(),
        one_way_share=round(one_way / max(1, sum(a != b for a, b in graph.edges)), 6),
        weak_components=len(components), largest_weak_component=max(components),
        isolated_nodes=nx.number_of_isolates(graph), route_arcs=len(route),
        route_breaks=int(np.sum(ordered[1:, 0] != ordered[:-1, 1])),
        nonincreasing_steps=int((dt <= 0).sum()), one_second_steps=int((dt == 1).sum()),
        gaps_over_30s=int((dt > 30).sum()), max_gap_s=float(max(dt, default=0)),
        speed_over_55_m_s=int((speed > 55).sum()), stationary_steps_under_0_5_m_s=int((speed < 0.5).sum()),
        max_speed_m_s=float(max(speed, default=0)),
        min_lon=float(valid_coord[:, 0].min()), max_lon=float(valid_coord[:, 0].max()),
        min_lat=float(valid_coord[:, 1].min()), max_lat=float(valid_coord[:, 1].max()),
        **files,
    )


def load_split(root, split):
    for folder in sorted((root / split).iterdir()):
        if folder.is_dir() and folder.name.isdecimal():
            trip = Trip.load(folder)
            yield folder, Geometry(trip), load_reference(folder, len(trip.arcs))


def tune_radius(root, output):
    results = []
    for folder, geometry, route in load_split(root, "train"):
        alignment = align_reference(geometry, route)
        hit = np.zeros(len(RADII), dtype=int)
        mean_count = np.zeros(len(RADII), dtype=int)
        for obs_idx, ids, distance, *_ in candidate_rows(geometry, RADII[-1], MAX_CANDIDATES):
            true_distance = distance[ids == route[alignment[obs_idx]]]
            if len(true_distance):
                hit += true_distance[0] <= np.array(RADII)
            mean_count += np.searchsorted(distance, RADII, side="right")
        n = int(geometry.valid.sum())
        for i, radius in enumerate(RADII):
            results.append(dict(trip_id=folder.name, radius_m=radius, coverage=hit[i] / n,
                                mean_candidates=mean_count[i] / n, observations=n, hits=int(hit[i])))
    write_records(output / "training_radius_sweep.csv", results)
    macro = {r: float(np.mean([a["coverage"] for a in results if a["radius_m"] == r])) for r in RADII}
    radius = next((r for r in RADII if macro[r] >= 0.95), max(RADII, key=macro.get))
    print(f"Train-only radius sweep: {macro}; using {radius} m", flush=True)
    return radius, macro


def write_records(path, records):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def export_trip(split, geometry, route, radius, features, references, map_path):
    alignment = align_reference(geometry, route)
    track = geometry.trip.track
    trip_id = geometry.trip.trip_id
    summary = dict(labelled=0, hits=0, empty=0, feature_rows=0, flagged=0)
    chosen = []
    last = None
    for i, ids, dist, fraction, direction, connects in candidate_rows(geometry, radius, MAX_CANDIDATES):
        position = alignment[i]
        arc = route[position]
        label_distance = geometry.measure(i, np.array([arc]))[0][0]
        low_quality = label_distance > 40
        route_break = position > 0 and geometry.trip.arcs[route[position - 1], 1] != geometry.trip.arcs[arc, 0]
        at_gap = np.isfinite(geometry.dt[i]) and not 0 < geometry.dt[i] <= 30
        flags = "|".join(name for name, condition in (("far_from_reference", low_quality),
                         ("reference_break", route_break), ("time_gap", at_gap)) if condition)
        summary["flagged"] += bool(flags)
        label = ids == arc
        miss = not np.any(label)
        summary["labelled"] += 1
        summary["hits"] += not miss
        summary["empty"] += len(ids) == 0
        summary["feature_rows"] += len(ids)
        key = (split, trip_id, i, track[i, 2])
        references.writerow((*key, int(arc), int(position), round(label_distance, 3),
                             flags, len(ids), int(miss)))
        speed = geometry.speed[i]
        dt = geometry.dt[i]
        features.writerows((*key, int(a), round(d, 3), round(f, 6), round(c, 6),
                            round(speed, 3) if np.isfinite(speed) else None,
                            round(dt, 3) if np.isfinite(dt) else None,
                            round(geometry.length[a], 3), rank, int(con), int(y))
                           for rank, (a, d, f, c, con, y) in enumerate(
                               zip(ids, dist, fraction, direction, connects, label), 1))
        if map_path:
            chosen.append((i, int(arc), float(label_distance)))
    summary["candidate_miss_rate"] = 1 - summary["hits"] / summary["labelled"]
    if map_path:
        write_map(map_path, geometry, route, chosen)
    return summary


def export_parquet(csv_path, parquet_path):
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    strings = {name: pa.string() for name in ("split", "trip_id")}
    integer_fields = ("observation_index", "arc_id", "distance_rank", "connects_previous_best", "label")
    types = {**strings, **{name: pa.int32() for name in integer_fields}}
    types.update({name: pa.float64() for name in FEATURES if name not in types})
    with gzip.open(csv_path, "rb") as raw:
        batches = pacsv.open_csv(raw, convert_options=pacsv.ConvertOptions(column_types=types))
        with pq.ParquetWriter(parquet_path, batches.schema, compression="zstd") as writer:
            for batch in batches:
                writer.write_batch(batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing train and student_test")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    radius, coverage = tune_radius(args.data, args.out)
    results = []
    csv_path = args.out / "features.csv.gz"
    with ExitStack() as stack:
        features = csv.writer(stack.enter_context(gzip.open(csv_path, "wt", newline="", encoding="utf-8", compresslevel=3)))
        references = csv.writer(stack.enter_context((args.out / "point_references.csv").open("w", newline="", encoding="utf-8")))
        features.writerow(FEATURES)
        references.writerow(REFERENCES)
        for split in ("train", "student_test"):
            for folder, geometry, route in load_split(args.data, split):
                printed = folder.name in {"00000075", "00000052"}
                map_path = args.out / f"alignment_{folder.name}.html" if printed else None
                metrics = inventory(folder, geometry, route)
                metrics.update(export_trip(split, geometry, route, radius, features, references, map_path))
                results.append(dict(split=split, trip_id=folder.name, **metrics))
            print(f"Exported {split}: {sum(r['split'] == split for r in results)} trips", flush=True)
    write_records(args.out / "trip_inventory.csv", results)
    (args.out / "candidate_config.json").write_text(json.dumps(
        dict(radius_m=radius, max_candidates=MAX_CANDIDATES, sweep_radii_m=RADII,
             train_mean_trip_coverage=coverage), indent=2), encoding="utf-8")
    try:
        export_parquet(csv_path, args.out / "features.parquet")
    except ModuleNotFoundError as error:
        if error.name != "pyarrow":
            raise
        print("PyArrow unavailable: features.csv.gz contains all candidate rows.", flush=True)


if __name__ == "__main__":
    main()
