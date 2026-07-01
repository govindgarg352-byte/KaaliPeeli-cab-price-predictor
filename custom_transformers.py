# =============================================================================
# custom_transformers.py — Shared custom sklearn transformers
#
# Lives in its OWN file (not inside preprocessing.py) deliberately. If a
# class is defined inside a script that gets run directly (e.g.
# `python preprocessing.py`), Python pickles instances of that class with a
# reference to the `__main__` module, not the importable module path — so
# unpickling later from a different script (e.g. train_models.py) fails with
# "Can't get attribute 'FrequencyEncoder' on <module '__main__'>", even with
# a correct import statement. Keeping it here, in a module that's only ever
# imported (never run as __main__), gives it one stable, consistent module
# path that both preprocessing.py and train_models.py can rely on.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes a categorical column by how often each category appears,
    AS A PROPER sklearn transformer — meaning .fit() only ever sees
    whatever data the surrounding Pipeline/ColumnTransformer hands it
    (the train fold), and .transform() reuses those fitted frequencies
    on test data. Avoids the leakage of computing frequencies on the
    full dataset before the train/test split.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freq_maps_ = {
            col: X[col].value_counts(normalize=True) for col in X.columns
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            # Unseen categories at test time map to 0 (never seen in training)
            X[col] = X[col].map(self.freq_maps_[col]).fillna(0)
        return X.values

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.freq_maps_.keys()])
