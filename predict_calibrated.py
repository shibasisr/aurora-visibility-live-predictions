"""
Live prediction with the camera-based range (camera_calibration.py) added as a
separate step. Wraps predict_core.predict_aurora_probability unchanged: its
output is returned as is ("probability" stays the raw model value, with the
live cloud cover) with the camera fields added.
"""
import pandas as pd

import camera_calibration
import predict_core

CALIBRATION_NOTE = (
    "Chance a camera would see aurora in your sky if clear; an upper bound for the eye."
)


def predict_aurora_probability_calibrated(lat, lon, dt=None):
    """predict_core.predict_aurora_probability plus:
    probability_raw_if_clear     raw model value at cloud_cover = 0
    camera_range_if_clear        (low, high) 25-75% spread across cameras, clear skies
    calibration_support          'camera' / 'extrapolated' / 'unreliable'
    Cloud is not folded into the range: read it from inputs["cloud_cover"].
    """
    out = predict_core.predict_aurora_probability(lat, lon, dt)
    if out["inputs"] is None:  # geomagnetic-equator shortcut in predict_core
        return {**out, "probability_raw_if_clear": 0.0, "camera_range_if_clear": (0.0, 0.0),
                "calibration_support": "extrapolated", "calibration_note": CALIBRATION_NOTE}

    bundle = predict_core._load_model()
    clear_row = pd.DataFrame([{**out["inputs"], "cloud_cover": 0.0}])
    p_clear = float(bundle["pipeline"].predict_proba(clear_row[bundle["featureslow"]])[:, 1][0])
    lo, hi = camera_calibration.camera_range(p_clear)
    sup = camera_calibration.support(out["mlat"])

    result = {
        **out,
        "probability_raw_if_clear": p_clear,
        "camera_range_if_clear": (float(lo), float(hi)),
        "calibration_support": sup,
        "calibration_note": CALIBRATION_NOTE,
    }
    if sup == "unreliable":
        result["low_confidence"] = True
        result["low_confidence_reason"] = (
            "Polar cap: the model doesn't work this far north."
        )
    return result
