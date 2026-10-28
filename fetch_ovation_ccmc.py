"""
Rebuilds ovation_camera_minutes.csv.gz: OVATION Prime electron energy flux at each camera minute, from the
runs archived by NASA CCMC (https://iswa.ccmc.gsfc.nasa.gov/iswa_data_tree/model/ionosphere/ovation_prime/e_data/).

The archive has one file every 30 minutes (HH:15, HH:45) on a magnetic local time x magnetic latitude grid
(0.25 h x 0.5 deg). Each camera minute takes the nearest file, the camera's magnetic latitude and its magnetic
local time (aacgmv2). About 12% of the half-hours are missing from the archive; those minutes are left out.
Downloads ~25,000 files (~4 GB) but keeps only what it needs. Needs aacgmv2 (pip install aacgmv2), which
the notebooks don't.
"""
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import aacgmv2
import numpy as np
import pandas as pd

BASE = "https://iswa.ccmc.gsfc.nasa.gov/iswa_data_tree/model/ionosphere/ovation_prime/e_data"
UA = {"User-Agent": "aurora-camera-validation/1.0"}


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode()
        except Exception as e:
            if getattr(e, "code", None) == 404:
                return None
            time.sleep(2 * (k + 1))
    return None


def parse(txt):
    """(mlt, mlat) -> flux; the file ends with a 'Totals:' block that isn't part of the grid."""
    lines = txt.splitlines()[1:]
    lines = lines[:next((k for k, l in enumerate(lines) if l.startswith("Totals")), len(lines))]
    a = np.loadtxt(lines)
    return {(int(round(m / 0.25)) % 96, round(lat * 2) / 2): f for m, lat, f in a}


cams = pd.read_csv("camera_minutes.csv.gz", usecols=["camera", "time", "mlat", "mlon"], parse_dates=["time"]).drop_duplicates(["camera", "time"])
cams["hour"] = cams["time"].dt.floor("h")
ch = cams.groupby(["camera", "hour"])["mlon"].first().reset_index()
ch["mlt0"] = [aacgmv2.convert_mlt(np.array([lon]), h.to_pydatetime())[0] for lon, h in zip(ch["mlon"], ch["hour"])]
cams = cams.merge(ch[["camera", "hour", "mlt0"]], on=["camera", "hour"])
cams["mlt"] = (cams["mlt0"] + cams["time"].dt.minute / 60) % 24
cams["stamp"] = cams["hour"] + pd.to_timedelta(np.where(cams["time"].dt.minute < 30, 15, 45), "m")
cams["j"] = (np.rint(cams["mlt"] / 0.25).astype(int) % 96)
cams["lat"] = (np.rint(cams["mlat"].abs() * 2) / 2)

stamps = pd.Series(sorted(cams["stamp"].unique()))
need = cams.groupby("stamp")[["j", "lat"]].apply(lambda g: set(zip(g["j"], g["lat"])))
values = {}  # (stamp, j, lat) -> flux, only the cells a camera needs
for month, ss in stamps.groupby(stamps.dt.strftime("%Y/%m")):
    listing = get(f"{BASE}/{month}/") or ""
    avail = set(re.findall(r'href="(\d{8}_\d{4}UT_Ovation_Prime_Eflux_e,all\.txt)"', listing))
    todo = [(s, s.strftime("%Y%m%d_%H%MUT_Ovation_Prime_Eflux_e,all.txt")) for s in ss]
    todo = [(s, n) for s, n in todo if n in avail]

    def one(sn):
        s, n = sn
        txt = get(f"{BASE}/{month}/{n}")
        try:
            g = parse(txt) if txt else None
        except Exception:
            g = None
        return s, ({k: g.get(k, np.nan) for k in need[s]} if g else None)

    with ThreadPoolExecutor(6) as ex:
        for s, g in ex.map(one, todo):
            if g:
                values.update({(s, *k): v for k, v in g.items()})
    print(month, f"{len(todo)} files", flush=True)

cams["ovation_eflux"] = [values.get(k, np.nan) for k in zip(cams["stamp"], cams["j"], cams["lat"])]
out = cams.dropna(subset=["ovation_eflux"])[["camera", "time", "ovation_eflux"]]
out["ovation_eflux"] = out["ovation_eflux"].round(3)
out.to_csv("ovation_camera_minutes.csv.gz", index=False)
print(f"{len(out):,} of {len(cams):,} camera minutes have an archived OVATION value")
