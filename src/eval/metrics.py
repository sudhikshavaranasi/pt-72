"""Shared candidate and observation metrics for training and route evaluation."""
import numpy as np


def candidate_metrics(labels, scores):
    predicted = scores >= 0.5
    labels = labels.astype(bool)
    tp = int(np.sum(labels & predicted))
    tn = int(np.sum(~labels & ~predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    clipped = np.clip(scores.astype(float), 1e-8, 1-1e-8)
    return dict(
        accuracy=(tp+tn)/len(labels), precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1),
        macro_f1=tp/max(2*tp+fp+fn,1)+tn/max(2*tn+fp+fn,1),
        log_loss=float(-np.mean(np.where(labels,np.log(clipped),np.log1p(-clipped)))),
        tn=tn, fp=fp, fn=fn, tp=tp,
    )


def top_arcs(point_rows, candidate_arcs, scores, point_count):
    """Return one argmax candidate arc per observation. Rows must be sorted by point."""
    result = np.full(point_count,-1,np.int32)
    if len(scores):
        starts = np.r_[0, np.flatnonzero(np.diff(point_rows)) + 1]
        sizes = np.diff(np.r_[starts,len(scores)])
        top = np.maximum.reduceat(scores,starts)
        winners = np.where(scores == np.repeat(top,sizes),np.arange(len(scores)),len(scores))
        result[point_rows[starts]] = candidate_arcs[np.minimum.reduceat(winners,starts)]
    return result


def observation_metrics(true_arc, true_in_cands, pred_arc, from_node, to_node, piece):
    """Score aligned points and consecutive distinct predicted arcs in each piece."""
    aligned = true_arc >= 0
    compare = ((piece[:-1] == piece[1:]) & (pred_arc[:-1] >= 0) &
               (pred_arc[1:] >= 0) & (pred_arc[:-1] != pred_arc[1:]))
    next_matches = (to_node[:-1] == from_node[1:]) & compare
    eligible = int(compare.sum())
    return dict(
        top1_accuracy=float(np.mean(true_arc[aligned] == pred_arc[aligned])) if aligned.any() else None,
        candidate_miss_rate=float(np.mean(~true_in_cands[aligned])) if aligned.any() else None,
        route_continuity=float(next_matches.sum()/eligible) if eligible else None,
        continuity_transitions=eligible, aligned_points=int(aligned.sum()),
        no_prediction_points=int(np.sum(aligned & (pred_arc < 0))),
    )
