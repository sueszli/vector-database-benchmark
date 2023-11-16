"""Module containing the integrations of the deepchecks.tabular package with the h2o autoML package."""
import numpy as np
import pandas as pd
try:
    import h2o
except ImportError as e:
    raise ImportError('H2OWrapper requires the h2o python package. To get it, run "pip install h2o".') from e

class H2OWrapper:
    """Deepchecks Wrapper for the h2o autoML package."""

    def __init__(self, h2o_model):
        if False:
            for i in range(10):
                print('nop')
        self.model = h2o_model

    def predict(self, df: pd.DataFrame) -> np.array:
        if False:
            print('Hello World!')
        'Predict the class labels for the given data.'
        return self.model.predict(h2o.H2OFrame(df)).as_data_frame().values[:, 0]

    def predict_proba(self, df: pd.DataFrame) -> np.array:
        if False:
            return 10
        'Predict the class probabilities for the given data.'
        return self.model.predict(h2o.H2OFrame(df)).as_data_frame().values[:, 1:].astype(float)

    @property
    def feature_importances_(self) -> np.array:
        if False:
            return 10
        'Return the feature importances based on h2o internal calculation.'
        try:
            return self.model.varimp(use_pandas=True)['percentage'].values
        except:
            return None