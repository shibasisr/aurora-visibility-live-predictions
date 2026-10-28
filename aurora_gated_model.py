"""
GatedAuroraModel -- blends the full EBM with logistic-regression fallbacks in regions
where the EBM's binned shape functions have too little training data to extrapolate
reliably (they flatline past the edge of what they've seen, rather than continuing a
trend). Two independent gates, each a smooth sigmoid weight rather than a hard cutoff:

1. oval_dist gate: trusts the EBM where `oval_dist` is well-populated (roughly higher
   magnetic latitude), hands off to LR(oval_dist, |mlat|) in the sparse low-mlat tail.

2. sun_elevation_utc gate: an LR(sun_elevation_utc) fallback for the sparse daylight
   tail; the rest is left to the EBM. A deep-dark fallback is optional (lr_sun_neg) and
   the deployed model doesn't use one: below about -18 deg the sky is fully dark, the EBM's
   curve is flat there as it should be, and a sun-only LR would pull deep-night
   predictions toward its own constant regardless of location.

Standalone module so joblib can pickle/unpickle it.
"""
import numpy as np


class GatedAuroraModel:
    def __init__(self, baseline_pipeline, lr_oval, featureslow, lr_features,
                 gate_center, gate_scale,
                 lr_sun_neg, lr_sun_pos,
                 sun_gate_neg_center, sun_gate_neg_scale,
                 sun_gate_pos_center, sun_gate_pos_scale):
        self.baseline_pipeline = baseline_pipeline
        self.lr_oval = lr_oval
        self.featureslow = featureslow
        self.lr_features = list(lr_features)
        self.gate_center = gate_center
        self.gate_scale = gate_scale
        self.lr_sun_neg = lr_sun_neg
        self.lr_sun_pos = lr_sun_pos
        self.sun_gate_neg_center = sun_gate_neg_center
        self.sun_gate_neg_scale = sun_gate_neg_scale
        self.sun_gate_pos_center = sun_gate_pos_center
        self.sun_gate_pos_scale = sun_gate_pos_scale

    def gate_weight(self, oval_dist):
        oval_dist = np.asarray(oval_dist, dtype=float)
        return 1.0 / (1.0 + np.exp(-(oval_dist - self.gate_center) / self.gate_scale))

    def sun_gate_weight(self, sun_elev):
        sun_elev = np.asarray(sun_elev, dtype=float)
        if self.lr_sun_neg is None:
            w_neg = np.ones_like(sun_elev)
        else:
            w_neg = 1.0 / (1.0 + np.exp(-(sun_elev - self.sun_gate_neg_center) / self.sun_gate_neg_scale))
        w_pos = 1.0 / (1.0 + np.exp((sun_elev - self.sun_gate_pos_center) / self.sun_gate_pos_scale))
        return np.where(sun_elev < 0, w_neg, w_pos)

    def _lr_input(self, X):
        df = X.copy()
        if "abs_mlat" in self.lr_features and "abs_mlat" not in df.columns:
            df["abs_mlat"] = df["mlat"].abs()
        return df[self.lr_features]

    def predict_proba(self, X):
        p_baseline = self.baseline_pipeline.predict_proba(X[self.featureslow])[:, 1]
        p_lr = self.lr_oval.predict_proba(self._lr_input(X))[:, 1]
        w = self.gate_weight(X["oval_dist"].values)
        p1 = w * p_baseline + (1 - w) * p_lr

        se = X["sun_elevation_utc"].values
        is_neg = se < 0
        p_pos = self.lr_sun_pos.predict_proba(X[["sun_elevation_utc"]])[:, 1]
        p_neg = p_pos if self.lr_sun_neg is None else self.lr_sun_neg.predict_proba(X[["sun_elevation_utc"]])[:, 1]
        p_lr_sun = np.where(is_neg, p_neg, p_pos)
        w_sun = self.sun_gate_weight(se)
        p1 = w_sun * p1 + (1 - w_sun) * p_lr_sun

        return np.column_stack([1 - p1, p1])
