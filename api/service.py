"""Request-time PathSnap inference and server-side evaluation."""
from functools import lru_cache
import logging
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from src._p1_data import SegmentIndex, load_reference
from src.eval.metrics import observation_metrics
from src.features import RADIUS_M, build_candidates
from src.geo import Geometry, Trip
from src.hmm import RoadGraph, decode, route_records
from src.labels import align_reference
from src.scoring import Scorer

TRIP_ID = re.compile(r"[0-9]{8}")
logger = logging.getLogger("pathsnap")


class PathSnap:
    def __init__(self, dataset_dir, *, model_path=None):
        self.directory = Path(dataset_dir).resolve()
        if not self.directory.is_dir():
            raise ValueError(f"Trip directory does not exist: {self.directory}")
        self.trip_dirs = {
            d.name: d for d in sorted(self.directory.iterdir())
            if d.is_dir() and TRIP_ID.fullmatch(d.name)
            and all((d / f"{d.name}.{ext}").is_file() for ext in ("track","nodes","arcs"))
        }
        if not self.trip_dirs:
            raise ValueError("No complete trip folders found")
        if len(self.trip_dirs) != 10:
            logger.warning("Loaded %d trip choices. The evaluation session expects ten.", len(self.trip_dirs))
        self.scorer = Scorer() if model_path is None else Scorer(model_path)
        self.load_context = lru_cache(maxsize=2)(self._load_context)

    def _load_context(self, trip_id):
        g = Geometry(Trip.load(self.trip_dirs[trip_id]))
        return g, SegmentIndex(g,RADIUS_M), RoadGraph(g)

    def predict(self, trip_id, *, comparison=False):
        folder = self.trip_dirs[trip_id]
        start = perf_counter()
        g, index, graph = self.load_context(trip_id)
        candidates = build_candidates(folder, geometry=g, index=index)
        candidates["score"] = self.scorer.score(candidates)
        result = decode(folder,candidates,geometry=g,graph=graph)
        inference_ms = round((perf_counter()-start)*1000,2)
        metrics = self.evaluate(folder,g,candidates,result["points"])
        response = dict(trip_id=trip_id,n_points=len(g.trip.track),
                        gps=g.trip.track[g.valid,:2][:,::-1].tolist(),
                        predicted_route=result["route"],metrics={**metrics,"inference_ms":inference_ms})
        if comparison:
            points = result["points"]
            response["classifier_route"] = route_records(
                g.trip, points["pred_arc_clf"], points["piece_id"]
            )
        return response

    @staticmethod
    def evaluate(folder,g,candidates,points):
        metric = dict(top1_acc_hmm=None,top1_acc_clf=None,route_continuity=None,candidate_miss_rate=None)
        route_file = folder / f"{folder.name}.route"
        if not route_file.is_file():
            return metric
        route = load_reference(folder,len(g.trip.arcs))
        positions = align_reference(g,route)
        truth = np.full(len(positions),-1,dtype=np.int32)
        valid = positions >= 0
        truth[valid] = route[positions[valid]]
        obs = candidates["obs_idx"].to_numpy(dtype=np.int32)
        cand_arc = candidates["arc_id"].to_numpy(dtype=np.int32)
        present = np.zeros(len(positions),dtype=bool)
        present[obs[truth[obs]==cand_arc]] = True
        piece = points["piece_id"].to_numpy()
        for column,key in (("pred_arc_clf","top1_acc_clf"),("pred_arc_hmm","top1_acc_hmm")):
            pred = points[column].to_numpy()
            has = pred >= 0
            start = np.full(len(pred),-1,dtype=np.int32)
            end = start.copy()
            start[has],end[has] = g.trip.arcs[pred[has]].T
            scores = observation_metrics(truth,present,pred,start,end,piece)
            metric[key] = scores["top1_accuracy"]
            if key == "top1_acc_hmm":
                metric["route_continuity"] = scores["route_continuity"]
                metric["candidate_miss_rate"] = scores["candidate_miss_rate"]
        return metric
