"""Feature transforms used in the model pipeline. Kept as importable, named
functions (not inline lambdas) so the fitted pipeline can be pickled."""
import numpy as np


def sine_mlon(x):
    return np.sin(x)


def cosine_mlon(x):
    return np.cos(x)
