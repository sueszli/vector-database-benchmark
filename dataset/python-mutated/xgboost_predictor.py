from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import pandas as pd
import xgboost
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.air.data_batch_type import DataBatchType
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
from ray.train.predictor import Predictor
from ray.train.xgboost import XGBoostCheckpoint
from ray.util.annotations import PublicAPI
if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

@PublicAPI(stability='beta')
class XGBoostPredictor(Predictor):
    """A predictor for XGBoost models.

    Args:
        model: The XGBoost booster to use for predictions.
        preprocessor: A preprocessor used to transform data batches prior
            to prediction.
    """

    def __init__(self, model: xgboost.Booster, preprocessor: Optional['Preprocessor']=None):
        if False:
            while True:
                i = 10
        self.model = model
        super().__init__(preprocessor)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}(model={self.model!r}, preprocessor={self._preprocessor!r})'

    @classmethod
    def from_checkpoint(cls, checkpoint: XGBoostCheckpoint) -> 'XGBoostPredictor':
        if False:
            i = 10
            return i + 15
        'Instantiate the predictor from a Checkpoint.\n\n        This is a helper constructor that instantiates the predictor from a\n        framework-specific XGBoost checkpoint.\n\n        Args:\n            checkpoint: The checkpoint to load the model and preprocessor from.\n\n        '
        model = checkpoint.get_model()
        preprocessor = checkpoint.get_preprocessor()
        return cls(model=model, preprocessor=preprocessor)

    def predict(self, data: DataBatchType, feature_columns: Optional[Union[List[str], List[int]]]=None, dmatrix_kwargs: Optional[Dict[str, Any]]=None, **predict_kwargs) -> DataBatchType:
        if False:
            while True:
                i = 10
        'Run inference on data batch.\n\n        The data is converted into an XGBoost DMatrix before being inputted to\n        the model.\n\n        Args:\n            data: A batch of input data.\n            feature_columns: The names or indices of the columns in the\n                data to use as features to predict on. If None, then use\n                all columns in ``data``.\n            dmatrix_kwargs: Dict of keyword arguments passed to ``xgboost.DMatrix``.\n            **predict_kwargs: Keyword arguments passed to ``xgboost.Booster.predict``.\n\n\n        Examples:\n\n        .. testcode::\n\n            import numpy as np\n            import xgboost as xgb\n            from ray.train.xgboost import XGBoostPredictor\n            train_X = np.array([[1, 2], [3, 4]])\n            train_y = np.array([0, 1])\n            model = xgb.XGBClassifier().fit(train_X, train_y)\n            predictor = XGBoostPredictor(model=model.get_booster())\n            data = np.array([[1, 2], [3, 4]])\n            predictions = predictor.predict(data)\n            # Only use first and second column as the feature\n            data = np.array([[1, 2, 8], [3, 4, 9]])\n            predictions = predictor.predict(data, feature_columns=[0, 1])\n\n        .. testcode::\n\n            import pandas as pd\n            import xgboost as xgb\n            from ray.train.xgboost import XGBoostPredictor\n            train_X = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])\n            train_y = pd.Series([0, 1])\n            model = xgb.XGBClassifier().fit(train_X, train_y)\n            predictor = XGBoostPredictor(model=model.get_booster())\n            # Pandas dataframe.\n            data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])\n            predictions = predictor.predict(data)\n            # Only use first and second column as the feature\n            data = pd.DataFrame([[1, 2, 8], [3, 4, 9]], columns=["A", "B", "C"])\n            predictions = predictor.predict(data, feature_columns=["A", "B"])\n\n\n        Returns:\n            Prediction result.\n\n        '
        return Predictor.predict(self, data, feature_columns=feature_columns, dmatrix_kwargs=dmatrix_kwargs, **predict_kwargs)

    def _predict_pandas(self, data: 'pd.DataFrame', feature_columns: Optional[Union[List[str], List[int]]]=None, dmatrix_kwargs: Optional[Dict[str, Any]]=None, **predict_kwargs) -> 'pd.DataFrame':
        if False:
            for i in range(10):
                print('nop')
        dmatrix_kwargs = dmatrix_kwargs or {}
        feature_names = None
        if TENSOR_COLUMN_NAME in data:
            data = data[TENSOR_COLUMN_NAME].to_numpy()
            data = _unwrap_ndarray_object_type_if_needed(data)
            if feature_columns:
                data = data[:, feature_columns]
        elif feature_columns:
            data = data[feature_columns].to_numpy()
            if all((isinstance(fc, str) for fc in feature_columns)):
                feature_names = feature_columns
        else:
            feature_columns = data.columns.tolist()
            data = data.to_numpy()
            if all((isinstance(fc, str) for fc in feature_columns)):
                feature_names = feature_columns
        if feature_names:
            dmatrix_kwargs['feature_names'] = feature_names
        matrix = xgboost.DMatrix(data, **dmatrix_kwargs)
        df = pd.DataFrame(self.model.predict(matrix, **predict_kwargs))
        df.columns = ['predictions'] if len(df.columns) == 1 else [f'predictions_{i}' for i in range(len(df.columns))]
        return df