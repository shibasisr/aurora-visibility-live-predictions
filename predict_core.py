"""
Core live-prediction logic: predict_aurora_probability(lat, lon). Used by
aurora_lookup.ipynb and aurora_globe.ipynb.

NOAA SWPC for current geomagnetic indices, Open-Meteo forecast for
cloud_cover, astral for sun_elevation_utc/moon_darkness, aacgmv2 for
mlat/mlon, the persisted EBM (via GatedAuroraModel) for the prediction.
"""
import datetime
import math
from pathlib import Path

import aacgmv2
import astral.moon
import joblib
import pandas as pd
import requests
from astral import Observer
from astral.sun import elevation as sun_elevation_fn

from aurora_transforms import cosine_mlon, sine_mlon  # noqa: F401 -- required for joblib unpickling
from aurora_gated_model import GatedAuroraModel  # noqa: F401 -- required for joblib unpickling
from fetch_live_geomagnetic import get_live_geomagnetic_indices

MODEL_PATH = Path(__file__).parent / "aurora_ebm_model_final_deployment.joblib"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_model_bundle = None


def _load_model():
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def _fetch_live_cloud_cover(lat, lon, dt):
    resp = requests.get(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "cloud_cover",
            "forecast_days": 3,
            "timezone": "UTC",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()["hourly"]
    times = [datetime.datetime.fromisoformat(t).replace(tzinfo=datetime.timezone.utc) for t in data["time"]]
    diffs = [abs((t - dt).total_seconds()) for t in times]
    idx = diffs.index(min(diffs))
    return data["cloud_cover"][idx]


def _moon_darkness(dt):
    phase_norm = astral.moon.phase(dt.date()) / 28.0
    return 1 - 2 * min(phase_norm, 1 - phase_norm)


def predict_aurora_probability(lat, lon, dt=None):
    """Predict P(aurora visible) at a geographic (lat, lon), for the current
    moment (dt=None) -- forecasting ahead is not yet supported (see project
    memory: Bz/solar-wind forecast has no validated public source)."""
    if dt is None:
        dt = datetime.datetime.now(datetime.timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    geo = get_live_geomagnetic_indices()

    mlat, mlon, _mlt = aacgmv2.get_aacgm_coord(lat, lon, 0, dt.replace(tzinfo=None))

    if math.isnan(mlat):
        # AACGM-v2 is mathematically undefined within roughly the innermost few degrees of the
        # geomagnetic (not geographic) equator -- which is itself offset from the geographic
        # equator and varies by longitude, since Earth's magnetic dipole is tilted and offset.
        # A location this close to the geomagnetic equator is, by the same physics, essentially
        # never going to see aurora, so this is answered directly rather than fed to the model.
        return {
            "probability": 0.0,
            "mlat": None,
            "mlon": None,
            "low_confidence": True,
            "low_confidence_reason": (
                "This location is too close to the geomagnetic equator for AACGM coordinates "
                "to be computed -- aurora is not observable this close to the magnetic equator, "
                "so the probability is reported as 0 directly rather than passed to the model."
            ),
            "inputs": None,
            "geomagnetic_data_age_seconds": geo["data_age_seconds"],
        }

    oval_dist = abs(mlat) - (66.0 - 2.0 * geo["kp"])

    sun_elev = sun_elevation_fn(Observer(latitude=lat, longitude=lon), dt)
    cloud_cover = _fetch_live_cloud_cover(lat, lon, dt)
    moon_darkness = _moon_darkness(dt)

    row = pd.DataFrame([{
        "mlat": mlat,
        "mlon": mlon,
        "kp": geo["kp"],
        "bz_gsm": geo["bz_gsm"],
        "solar_wind_speed": geo["solar_wind_speed"],
        "oval_dist": oval_dist,
        "Ap": geo["ap_kp_implied"],
        "cloud_cover": cloud_cover,
        "moon_darkness": moon_darkness,
        "sun_elevation_utc": sun_elev,
    }])

    bundle = _load_model()
    prob = float(bundle["pipeline"].predict_proba(row[bundle["featureslow"]])[:, 1][0])

    low_confidence = abs(mlat) < 48 and geo["kp"] < 7

    return {
        "probability": prob,
        "mlat": mlat,
        "mlon": mlon,
        "low_confidence": low_confidence,
        "low_confidence_reason": (
            "mlat<48 without an active G4/G5 storm (kp<7) -- "
            "aurora unlikely, the probability estimates are unreliable"
            if low_confidence else None
        ),
        "inputs": row.iloc[0].to_dict(),
        "geomagnetic_data_age_seconds": geo["data_age_seconds"],
    }
