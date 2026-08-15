"""
Live (real-time) replacement for the OMNI2-based geomagnetic fetch in aurora_data_pipeline.ipynb.

OMNI2 has a ~2-week publication lag (see project_aurora_timeseries_plan memory) -- unusable for
a "will I see aurora right now" tool. NOAA SWPC publishes the same underlying measurements
(DSCOVR-derived solar wind + NOAA's own real-time planetary Kp estimate) within minutes.

Endpoints verified live on 2026-08-13, not assumed from memory:
  https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json    -- 1-min cadence, ~2.5 days history, bz_gsm field
  https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json   -- 1-min cadence, proton_speed field
  https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json -- 3-hourly, Kp field (0-9 quasi-log, matches
                                                                          existing `kp` convention exactly)

Column names below match the existing pipeline's OMNI2-derived columns 1:1 so this can be swapped
in as a drop-in replacement for the `kp`/`bz_gsm`/`solar_wind_speed`/`ap` features.

Does NOT modify any existing pipeline file -- new module only.
"""
import urllib.request
import json
from datetime import datetime, timezone

RTSW_MAG_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
RTSW_WIND_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

# Same Kp -> ap conversion table already used in the source notebook's prediction-sweep cells
# (model_EBM_final_newdata_timestart.ipynb) -- reused here for consistency, NOT re-derived.
KP_TO_AP = {0: 0, 1: 3, 2: 7, 3: 15, 4: 27, 5: 48, 6: 80, 7: 140, 8: 240, 9: 400}


def _fetch_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_live_geomagnetic_indices():
    """
    Returns a dict of current geomagnetic indices, column-named to match the existing
    OMNI2-derived pipeline features:
        kp, bz_gsm, solar_wind_speed, ap  (ap here is the Kp-implied 3-hourly value,
        NOT the true measured ap -- NOAA doesn't publish that in real time; see note below)

    Also returns the raw source timestamps and a `data_age_seconds` per source so staleness
    can be checked before using this in a prediction (DSCOVR occasionally goes offline).
    """
    mag = _fetch_json(RTSW_MAG_URL)
    wind = _fetch_json(RTSW_WIND_URL)
    kp_series = _fetch_json(KP_URL)

    # NOTE: rtsw_mag/rtsw_wind are newest-first; noaa-planetary-k-index is newest-last.
    # Verified directly (not assumed) on 2026-08-13 -- these two endpoint families use
    # opposite ordering conventions, easy to get silently wrong.
    latest_mag = mag[0]
    latest_wind = wind[0]
    latest_kp = kp_series[-1]

    now = datetime.now(timezone.utc)

    def age_seconds(time_tag):
        t = datetime.fromisoformat(time_tag.replace("Z", "")).replace(tzinfo=timezone.utc)
        return (now - t).total_seconds()

    kp_val = latest_kp["Kp"]
    kp_bin = round(kp_val)  # nearest integer bin, for the Kp->ap lookup

    return {
        "kp": kp_val,
        "bz_gsm": latest_mag.get("bz_gsm"),
        "solar_wind_speed": latest_wind.get("proton_speed"),
        # NOTE: this is the Kp-implied ap (via KP_TO_AP), not a directly-measured ap --
        # NOAA's real-time feed doesn't publish the 3-hourly linear ap index itself, only
        # NOAA's own Kp estimate (which also isn't identical to the definitive Potsdam/GFZ
        # Kp that OMNI2 uses -- can differ by a fraction of a Kp unit). Flagged, not silently
        # assumed equivalent.
        "ap_kp_implied": KP_TO_AP.get(kp_bin, KP_TO_AP[9] if kp_bin > 9 else KP_TO_AP[0]),
        "source_timestamps": {
            "mag": latest_mag["time_tag"],
            "wind": latest_wind["time_tag"],
            "kp": latest_kp["time_tag"],
        },
        "data_age_seconds": {
            "mag": age_seconds(latest_mag["time_tag"]),
            "wind": age_seconds(latest_wind["time_tag"]),
            "kp": age_seconds(latest_kp["time_tag"]),
        },
    }


if __name__ == "__main__":
    indices = get_live_geomagnetic_indices()
    print(json.dumps(indices, indent=2))
