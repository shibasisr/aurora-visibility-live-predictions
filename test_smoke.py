"""
Offline smoke test after cloning: python test_smoke.py (pytest also works).
Fixed inputs, no live data, a few seconds.
"""
import joblib
import numpy as np
import pandas as pd

import camera_calibration

BUNDLE = joblib.load("aurora_ebm_model_final_deployment.joblib")
MODEL, FEATURES = BUNDLE["pipeline"], BUNDLE["featureslow"]


def row(mlat, kp, sun, cloud=0.0, mlon=90.0, bz=-3.0, v=450.0, ap=15.0, moon=1.0):
    return {"mlat": mlat, "mlon": mlon, "kp": kp, "bz_gsm": bz, "solar_wind_speed": v,
            "oval_dist": abs(mlat) - (66 - 2 * kp), "Ap": ap, "cloud_cover": cloud, "moon_darkness": moon,
            "sun_elevation_utc": sun}


CASES = {  # name: (inputs, probability expected from the saved model)
    "active_night_60": (row(60, 5, -25), 0.9217),
    "quiet_night_60": (row(60, 1, -25, bz=1, v=350, ap=4), 0.4815),
    "quiet_deep_night_32": (row(32, 1, -62, bz=1, v=350, ap=4), 0.0207),
    "daylight_60": (row(60, 5, 30), 0.1130),
    "overcast_active_60": (row(60, 5, -25, cloud=100), 0.9082),
}


def predict(name):
    return MODEL.predict_proba(pd.DataFrame([CASES[name][0]])[FEATURES])[:, 1][0]


def test_fixed_inputs_give_saved_outputs():
    for name, (_, expected) in CASES.items():
        assert abs(predict(name) - expected) < 0.01, name


def test_physical_sanity():
    assert predict("active_night_60") > predict("quiet_night_60")
    assert predict("overcast_active_60") <= predict("active_night_60")
    assert predict("daylight_60") < predict("active_night_60")
    assert predict("quiet_deep_night_32") < 0.05


def test_camera_range():
    p = np.linspace(0.001, 0.999, 50)
    lo, hi = camera_calibration.camera_range(p)
    assert np.all((0 <= lo) & (lo <= hi) & (hi <= 1))
    assert np.all(np.diff(lo) >= 0) and np.all(np.diff(hi) >= 0)


def test_support_labels():
    assert camera_calibration.support(55) == "camera"
    assert camera_calibration.support(-55) == "extrapolated"
    assert camera_calibration.support(66) == "extrapolated"
    assert camera_calibration.support(76) == "unreliable"
    assert camera_calibration.support(-76) == "unreliable"


def test_data_files():
    for f, cols in [("camera_minutes.csv.gz", {"camera", "time", "mlat", "kp", "aurora"}),
                    ("svalbard_minutes.csv.gz", {"time", "mlat", "kp", "aurora"}),
                    ("ovation_camera_minutes.csv.gz", {"camera", "time", "ovation_eflux"})]:
        d = pd.read_csv(f)
        assert cols <= set(d.columns), f
        if "camera" in d:
            assert not d.duplicated(["camera", "time"]).any(), f


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
