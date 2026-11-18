"""Score label-free candidate DataFrames with the exported model."""
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_model(path=ROOT / "models" / "final_model.joblib"):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


class Scorer:
    def __init__(self, path=ROOT / "models" / "final_model.joblib"):
        self.model = load_model(path)

    def score(self, df):
        return self.model.probability(df.loc[:, list(self.model.feature_cols)].to_numpy(dtype=np.float32))


_default = None


def score(df):
    global _default
    if _default is None:
        _default = Scorer()
    return _default.score(df)
