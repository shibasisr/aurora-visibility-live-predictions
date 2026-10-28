# Model, calibration and limitations

The details behind [README.md](README.md).

## Model

An Explainable Boosting Machine (EBM / GA2M, via [InterpretML](https://github.com/interpretml/interpret))
on: magnetic latitude and longitude, Kp, Bz (GSM), solar wind speed, Ap, cloud cover, moon darkness, sun
elevation, and `oval_dist` (distance from the Feldstein-Starkov auroral oval boundary — see a simple
empirical formula [here](https://www.spaceweather.gov/content/tips-viewing-aurora)), plus three interaction
terms: Kp × Bz, Kp × solar wind speed and Bz × solar wind speed.

EBM's shape functions flatline beyond the edge of their training data. Therefore extrapolation requires care. Two features
have sparse regions where this caused implausible predictions (garbage high values at very low magnetic
latitude, where the true probability should be near zero; an unphysical "recovery" at very high sun
elevation). Both are fixed with a region-gated
fallback: a smooth weight trusts the full EBM wherever training data is dense, and switches to a simple
logistic regression (continuous, so it degrades to a physically consistent value instead of flatlining) only in the genuinely
sparse tails.
See `aurora_gated_model.py` for the implementation.

## Calibration

The raw percentages aren't literal, so the calibrated notebooks add a camera-based range next to them. The
evidence behind it is in [`CAMERA_VALIDATION.md`](CAMERA_VALIDATION.md); this is what the code does:

1. The model is run with cloud cover set to 0 (clear sky). Cloud is shown separately, not folded in.
2. That probability goes through nine per-camera maps, `sigmoid(a * logit(p) + b)`, fitted in
   `camera_calibration.ipynb` and saved to `camera_calibration_maps.json`.
3. The range shown is the 25-75% spread of the nine results (`camera_calibration.py`).
4. Each location gets a support label: `camera` for northern observers at ~51-60° mlat (the cameras are all north), `extrapolated`
   elsewhere, including the whole southern hemisphere,
   and `unreliable` above ~72°, where predictions are flagged low-confidence.

`probability` from `predict_core` is unchanged; the camera fields come from `predict_calibrated.py`.

## Known limitations

The citizen-science data suffers from reporting bias leading to class imbalance: observations are sparse
overall, and the concentration of observations from specific locations suggests viewing hotspots rather
than a representative sample. Sample weights are rebalanced per magnetic-latitude band
(`00-48`, `48-63`, `63-90`) to partially correct for this, but the model is still stretched in a couple of
scenarios:

- **Low Kp (calm conditions), `mlat 63-90`**: the model predicts a high viewing chance regardless of Kp,
  because the citizen labels are ~87% positive there. Some of this is real: at 65-67° all-sky cameras record
  aurora overhead in 24-49% of quiet (Kp 0-1) minutes, and a person sees more of the sky than a camera's
  overhead view. But the cameras still rise steeply with Kp (Tromsø 24% → 92% from Kp 0-1 to Kp 4),
  structure the citizen data has lost and that only more negative-sighting reports from this regime could
  recover. Further north, in the polar cap (Svalbard, 76°), aurora is common at any Kp and the model has no
  skill there (flagged unreliable above ~72°).
- **Cloud cover**: one of the weakest terms in the model (going from 0% to 100% cloud lowers the average
  prediction only from ~60% to ~57%). Two reasons: people mostly go out when the weather looks good (54% of
  reports come from skies with ≤10% cloud), and the historical cloud value is Open-Meteo's hourly archive
  estimate on a coarse grid, which often won't match the observer's sky (67% of reports under 90-100% cloud are
  still "seen"). The calibrated lookup shows live cloud separately for this reason.
- **Southern hemisphere**: only 2.9% of training rows (647/22,174) come from south of the magnetic equator. The
  model's own shape function scores the extreme southern latitudes higher than anywhere in the
  well-populated north — an artifact of that sparsity, not a real hemispheric asymmetry, since the model
  has no feature (e.g. IMF `By`, season) that could actually capture real interhemispheric differences.
  Southern-hemisphere predictions are correspondingly unreliable.
