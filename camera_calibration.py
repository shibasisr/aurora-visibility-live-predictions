"""
Camera-based range for the model's probability -- a separate step applied after
the model (aurora_ebm_model_final_deployment.joblib is unchanged).

The maps are fitted in camera_calibration.ipynb and saved to
camera_calibration_maps.json. In short: the model's latitude behaviour matches
an all-sky camera ~7 deg magnetic latitude poleward of the observer, so an
observer at |mlat| L is compared with clean all-sky camera labels at L + 7
(6 THEMIS cameras at 58-63 deg, Tromso / Kiruna / Skibotn at 65-67 deg; clear,
dark, moon-free minutes). For each camera alone the map
sigmoid(a * logit(p) + b) is fitted; the range shown is the 25-75% spread of
those nine maps at the model's clear-sky probability. Each map is monotone, so
the ranking of predictions is unchanged.

Why a range and not one number: the cameras disagree in level, and no single
map beats the raw model on average across them (see the notebook).

Meaning: chance that aurora a camera would detect is in your sky, including low
on the northern horizon -- roughly an upper bound for the naked eye. Clear skies
only (cloud_cover = 0). Camera-backed for northern observers at ~51-60 deg mlat (all the cameras are north);
elsewhere the maps are applied without camera backing; above ~72 deg
predictions are unreliable (Svalbard all-sky camera, AUC 0.55).
"""
import json
from pathlib import Path

import numpy as np

_MAPS = json.loads((Path(__file__).parent / "camera_calibration_maps.json").read_text())
PER_CAMERA = {c: (m["a"], m["b"]) for c, m in _MAPS["cameras"].items()}  # (a, b) of the map fitted to one camera alone
SUPPORTED_MLAT = tuple(_MAPS["supported_mlat"])
UNRELIABLE_MLAT = _MAPS["unreliable_mlat"]
RANGE_QUANTILES = tuple(_MAPS["range_quantiles"])
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def camera_range(p, q=RANGE_QUANTILES):
    """(low, high): spread of the per-camera maps at p; p a float or a numpy array."""
    z = _logit(p)
    vals = np.array([1 / (1 + np.exp(-(a * z + b))) for a, b in PER_CAMERA.values()])
    lo, hi = np.quantile(vals, q, axis=0)
    return lo, hi


def support(mlat):
    """'camera' (northern observers ~51-60 deg), 'extrapolated', or 'unreliable' (|mlat| above ~72 deg).
    mlat is signed: the cameras are all in the north, so no southern location is camera-backed."""
    if abs(mlat) > UNRELIABLE_MLAT:
        return "unreliable"
    if SUPPORTED_MLAT[0] <= mlat <= SUPPORTED_MLAT[1]:
        return "camera"
    return "extrapolated"


# what each label means for the person looking it up (the camera that matters is ~7 deg north of them)
SUPPORT_TEXT = {
    "camera": "camera-backed: a camera ~7° north of you checks the model",
    "extrapolated": "not camera-backed: no camera ~7° north of you",
    "unreliable": "unreliable: polar cap, where the Svalbard camera shows the model fails",
}
