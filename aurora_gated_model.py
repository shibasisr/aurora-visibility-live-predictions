"""
GatedAuroraModel: the EBM blended with logistic-regression fallbacks where its shape functions run out of
training data. Two smooth sigmoid gates:

1. oval_dist: LR(oval_dist, |mlat|) in the sparse low-mlat tail.
2. sun_elevation_utc: LR(sun_elevation_utc) in the sparse daylight tail. A deep-dark fallback (lr_sun_neg) is
   optional; the deployed model has none.

Own module so joblib can pickle it.
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
