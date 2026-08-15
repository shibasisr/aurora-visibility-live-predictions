# Aurora Visibility Prediction

Live prediction of aurora visibility probability at any location, using real-time [NOAA SWPC geomagnetic
data](https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json) rather than [OMNI2](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/). OMNI2's ~2-week publication lag makes it unusable for live queries, so it's used
only for historical training data. Trained on ~22,000 Aurorasaurus citizen-science sightings (2014-2025) [data](https://zenodo.org/records/16783265) merged with
historical geomagnetic and weather data.

This is a continuation of the
[summer26-aurora-visibility-predictions](https://github.com/Erdos-Projects/summer26-aurora-visibility-predictions)
project, where the initial modelling was done. This version builds on those findings and further refines
the predictions. Forecasts at a 6-36 hour horizon will be available soon.

## What's here

- `predict_core.py` — core prediction function (`predict_aurora_probability(lat, lon)`). Fetches live
  kp/Bz/solar wind from NOAA SWPC, live cloud cover from Open-Meteo, computes magnetic coordinates and
  sun elevation/moon darkness, and runs them through the model to get a probability.
- `aurora_lookup.ipynb` — interactive widget: takes a location (lat, lon), outputs a live probability.
- `aurora_globe.ipynb` — interactive orthographic globe, live probability across the northern hemisphere.
  Not reliable for southern hemisphere predictions, although the option to view them exists (see Known
  limitations).
- `aurora_ebm_model_final_deployment.joblib` — the trained model (see Model below).
- `aurora_gated_model.py`, `aurora_transforms.py` — supporting classes/functions required to unpickle and
  run the model.
- `fetch_live_geomagnetic.py` — live NOAA SWPC data fetch.
- `aurora_final_deployment.ipynb` — trains the model on `aurora_dataset_clean_timestart.csv` and saves
  it to `aurora_ebm_model_final_deployment.joblib`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Run `aurora_lookup.ipynb` or `aurora_globe.ipynb` directly — both load the saved model, no retraining
needed. To retrain from scratch, run `aurora_final_deployment.ipynb`.

## Model

An Explainable Boosting Machine (EBM / GA2M, via [InterpretML](https://github.com/interpretml/interpret))
on: magnetic latitude (mlat), kp, Bz (GSM), solar wind speed, Ap, cloud cover, moon darkness, sun elevation, and
`oval_dist` (distance from the Feldstein-Starkov auroral oval boundary — see a simple empirical formula
[here](https://www.spaceweather.gov/content/tips-viewing-aurora)).

EBM's shape functions flatline beyond the edge of their training data. Therefore extrapolation requires care. Two features
have sparse regions where this caused implausible predictions (garbage high values at very low magnetic
latitude, where the true probability should be near zero; an unphysical "recovery" at very high sun
elevation). Both are fixed with a region-gated
fallback: a smooth weight trusts the full EBM wherever training data is dense, and switches to a simple
logistic regression (continuous, so it degrades to a physically consistent value instead of flatlining) only in the genuinely
sparse tails.
See `aurora_gated_model.py` for the implementation.

## Known limitations

The citizen-science data suffers from reporting bias leading to class imbalance: observations are sparse
overall, and the concentration of observations from specific locations suggests viewing hotspots rather
than a representative sample. Sample weights are rebalanced per magnetic-latitude band
(`00-48`, `48-63`, `63-90`) to partially correct for this, but the model is still stretched in a couple of
scenarios:

- **Low kp (calm conditions), `mlat 63-90`**: the model essentially ignores the geomagnetic indices and
  predicts a high viewing chance regardless. This isn't a bug — the training labels themselves are ~88%
  positive in this regime, so the model has correctly learned the data it was given. The finer,
  kp-dependent structure is genuinely lost, though, and can only be recovered with more negative-sighting
  reports from this regime.
- **Southern hemisphere**: only 2.9% of training rows (647/22,280) come from south of the equator. The
  model's own shape function scores the extreme southern latitudes higher than anywhere in the
  well-populated north — an artifact of that sparsity, not a real hemispheric asymmetry, since the model
  has no feature (e.g. IMF `By`, season) that could actually capture real interhemispheric differences.
  Southern-hemisphere predictions are correspondingly unreliable.
