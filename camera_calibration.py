"""
Camera-based range for Model 1's clear-sky probability, applied after the model.

Nine per-camera maps sigmoid(a * logit(p) + b), fitted in camera_calibration.ipynb and read from
camera_calibration_maps.json; the range is their 25-75% spread at p. Background: CAMERA_VALIDATION.md.
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
    "camera": "checked against a camera",
    "extrapolated": "no camera check for your latitude",
    "unreliable": "unreliable",
}
