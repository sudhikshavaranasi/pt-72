"""Infer monotone GPS-to-route labels from the ordered reference arcs."""
import numpy as np


def align_reference(geometry, route):
    """Minimize distance, heading mismatch and a soft skipped-arc cost.

    Route positions can stay the same or move forward. The original positions
    are preserved even if one arc ID occurs more than once in the route.
    """
    observations = np.flatnonzero(geometry.valid)
    positions = np.arange(len(route), dtype=np.int32)
    parent = np.empty((len(observations), len(route)), dtype=np.int32)
    cost = 2.0 * positions
    for step, obs_idx in enumerate(observations):
        distance, _, direction = geometry.measure(obs_idx, route)
        emission = distance + 8.0 * (1 - direction)
        if step:
            gap = geometry.trip.track[obs_idx, 2] - geometry.trip.track[observations[step - 1], 2]
            skip_cost = 0.15 if gap > 30 else 2.0
            adjusted = cost - skip_cost * positions
            prefix = np.minimum.accumulate(adjusted)
            previous = np.maximum.accumulate(np.where(adjusted == prefix, positions, 0))
            parent[step] = previous
            cost = emission + prefix + skip_cost * positions
        else:
            cost = emission + cost
    position = int(np.argmin(cost + 2.0 * (len(route) - 1 - positions)))
    result = np.full(len(geometry.gps), -1, dtype=np.int32)
    for step in range(len(observations) - 1, -1, -1):
        result[observations[step]] = position
        if step:
            position = parent[step, position]
    return result
