from typing import TYPE_CHECKING, List, Optional, Union
import lightgbm
import pandas as pd
from pandas.api.types import is_object_dtype
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.air.data_batch_type import DataBatchType
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
from ray.train.lightgbm import LightGBMCheckpoint
from ray.train.predictor import Predictor
from ray.util.annotations import PublicAPI
if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

@PublicAPI(stability='beta')
class LightGBMPredictor(Predictor):
    """A predictor for LightGBM models.

    Args:
        model: The LightGBM booster to use for predictions.
        preprocessor: A preprocessor used to transform data batches prior
            to prediction.
    """

    def __init__(self, model: lightgbm.Booster, preprocessor: Optional['Preprocessor']=None):
        if False:
            for i in range(10):
                print('nop')
        self.model = model
        super().__init__(preprocessor)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}(model={self.model!r}, preprocessor={self._preprocessor!r})'

    @classmethod
    def from_checkpoint(cls, checkpoint: LightGBMCheckpoint) -> 'LightGBMPredictor':
        if False:
            return 10
        'Instantiate the predictor from a LightGBMCheckpoint.\n\n        Args:\n            checkpoint: The checkpoint to load the model and preprocessor from.\n\n        '
        model = checkpoint.get_model()
        preprocessor = checkpoint.get_preprocessor()
        return cls(model=model, preprocessor=preprocessor)

    def predict(self, data: DataBatchType, feature_columns: Optional[Union[List[str], List[int]]]=None, **predict_kwargs) -> DataBatchType:
        if False:
            for i in range(10):
                print('nop')
        'Run inference on data batch.\n\n        Args:\n            data: A batch of input data.\n            feature_columns: The names or indices of the columns in the\n                data to use as features to predict on. If None, then use\n                all columns in ``data``.\n            **predict_kwargs: Keyword arguments passed to\n                ``lightgbm.Booster.predict``.\n\n        Examples:\n            >>> import numpy as np\n            >>> import lightgbm as lgbm\n            >>> from ray.train.lightgbm import LightGBMPredictor\n            >>>\n            >>> train_X = np.array([[1, 2], [3, 4]])\n            >>> train_y = np.array([0, 1])\n            >>>\n            >>> model = lgbm.LGBMClassifier().fit(train_X, train_y)\n            >>> predictor = LightGBMPredictor(model=model.booster_)\n            >>>\n            >>> data = np.array([[1, 2], [3, 4]])\n            >>> predictions = predictor.predict(data)\n            >>>\n            >>> # Only use first and second column as the feature\n            >>> data = np.array([[1, 2, 8], [3, 4, 9]])\n            >>> predictions = predictor.predict(data, feature_columns=[0, 1])\n\n            >>> import pandas as pd\n            >>> import lightgbm as lgbm\n            >>> from ray.train.lightgbm import LightGBMPredictor\n            >>>\n            >>> train_X = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])\n            >>> train_y = pd.Series([0, 1])\n            >>>\n            >>> model = lgbm.LGBMClassifier().fit(train_X, train_y)\n            >>> predictor = LightGBMPredictor(model=model.booster_)\n            >>>\n            >>> # Pandas dataframe.\n            >>> data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])\n            >>> predictions = predictor.predict(data)\n            >>>\n            >>> # Only use first and second column as the feature\n            >>> data = pd.DataFrame([[1, 2, 8], [3, 4, 9]], columns=["A", "B", "C"])\n            >>> predictions = predictor.predict(data, feature_columns=["A", "B"])\n\n\n        Returns:\n            Prediction result.\n\n        '
        return Predictor.predict(self, data, feature_columns=feature_columns, **predict_kwargs)

    def _predict_pandas(self, data: 'pd.DataFrame', feature_columns: Optional[Union[List[str], List[int]]]=None, **predict_kwargs) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        feature_names = None
        if TENSOR_COLUMN_NAME in data:
            data = data[TENSOR_COLUMN_NAME].to_numpy()
            data = _unwrap_ndarray_object_type_if_needed(data)
            if feature_columns:
                data = data[:, feature_columns]
            data = pd.DataFrame(data, columns=feature_names)
            data = data.infer_objects()
            update_dtypes = {}
            for column in data.columns:
                dtype = data.dtypes[column]
                if is_object_dtype(dtype):
                    update_dtypes[column] = pd.CategoricalDtype()
            if update_dtypes:
                data = data.astype(update_dtypes, copy=False)
        elif feature_columns:
            data = data[feature_columns]
        df = pd.DataFrame(self.model.predict(data, **predict_kwargs))
        df.columns = ['predictions'] if len(df.columns) == 1 else [f'predictions_{i}' for i in range(len(df.columns))]
        return df