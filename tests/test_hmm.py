import unittest
from types import SimpleNamespace

import numpy as np

from src.hmm import RoadGraph, Step, transition_log, viterbi


CONFIG = {
    "beta_m": 1.0,
    "beta_m_per_s": 0.0,
    "emission_weight": 0.1,
    "gap_threshold_s": 30,
    "stationary_m": 3.0,
    "stationary_max_dt_s": 3.0,
    "stationary_switch_cost": 3.0,
    "speed_spike_m_s": 55.0,
    "spike_beta_multiplier": 2.5,
    "road_slack_m": 50.0,
    "road_slack_m_s": 32.0,
    "max_road_m": 1200.0,
    "backtrack_tolerance_m": 10.0,
}


class TestRoadGraph(unittest.TestCase):

    def test_graph_preserves_direction(self):
        geometry = SimpleNamespace(
            nodes=np.zeros((3, 2)),
            trip=SimpleNamespace(
                arcs=np.array([[0, 1], [1, 2], [2, 1]])
            ),
            length=np.array([5.0, 7.0, 7.0])
        )

        graph = RoadGraph(geometry)

        forward = graph.distances(0, 100)
        backward = graph.distances(2, 100)

        self.assertEqual(forward[2], 12.0)
        self.assertNotIn(0, backward)


class TestViterbi(unittest.TestCase):

    def test_viterbi_uses_transition_costs(self):
        states = [
            (np.array([0, 1]), np.array([0.99, 0.01]),
             np.array([0.5, 0.5])),
            (np.array([2, 3]), np.array([0.01, 0.99]),
             np.array([0.5, 0.5])),
        ]

        step = Step(
            network_m=np.array([[0.0, 100.0],
                                [100.0, 100.0]]),
            observed_m=0.0,
            dt=1.0,
            stationary=False,
            speed_spike=False,
            same_arc=np.zeros((2, 2), dtype=bool)
        )

        clf, hmm, pieces, _ = viterbi(
            (states, [None, step]), CONFIG
        )

        self.assertEqual(clf.tolist(), [0, 3])
        self.assertEqual(hmm.tolist(), [0, 2])
        self.assertEqual(pieces.tolist(), [0, 0])

    def test_empty_candidates_split_sequence(self):
        states = [
            (np.array([0]), np.array([0.9]), np.array([0.5])),
            (np.array([], dtype=int), np.array([]), np.array([])),
            (np.array([1]), np.array([0.8]), np.array([0.5])),
        ]

        clf, hmm, pieces, splits = viterbi(
            (states, [None, None, None]), CONFIG
        )

        self.assertEqual(hmm.tolist(), [0, -1, 1])
        self.assertEqual(pieces.tolist(), [0, -1, 1])
        self.assertEqual(splits["zero_candidates"], 1)

    def test_unavailable_transition_starts_new_piece(self):
        states = [
            (np.array([0]), np.array([0.9]), np.array([0.5])),
            (np.array([1]), np.array([0.8]), np.array([0.5])),
        ]

        _, hmm, pieces, splits = viterbi(
            (states, [None, None]), CONFIG
        )

        self.assertEqual(hmm.tolist(), [0, 1])
        self.assertEqual(pieces.tolist(), [0, 1])
        self.assertEqual(splits["gap_or_clock"], 1)

    def test_stationary_transition_penalizes_switching(self):
        step = Step(
            network_m=np.array([[0.0, 10.0]]),
            observed_m=0.0,
            dt=1.0,
            stationary=True,
            speed_spike=False,
            same_arc=np.array([[True, False]])
        )

        result = transition_log(step, CONFIG)

        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[0, 1], -13.0)


if __name__ == "__main__":
    unittest.main()
