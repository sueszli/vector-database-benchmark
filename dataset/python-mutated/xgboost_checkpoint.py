import os
import tempfile
from typing import TYPE_CHECKING, Optional
import xgboost
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI
if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

@PublicAPI(stability='beta')
class XGBoostCheckpoint(FrameworkCheckpoint):
    """A :py:class:`~ray.train.Checkpoint` with XGBoost-specific functionality."""
    MODEL_FILENAME = 'model.json'

    @classmethod
    def from_model(cls, booster: xgboost.Booster, *, preprocessor: Optional['Preprocessor']=None) -> 'XGBoostCheckpoint':
        if False:
            return 10
        'Create a :py:class:`~ray.train.Checkpoint` that stores an XGBoost\n        model.\n\n        Args:\n            booster: The XGBoost model to store in the checkpoint.\n            preprocessor: A fitted preprocessor to be applied before inference.\n\n        Returns:\n            An :py:class:`XGBoostCheckpoint` containing the specified ``Estimator``.\n\n        Examples:\n\n            ... testcode::\n\n                import numpy as np\n                import ray\n                from ray.train.xgboost import XGBoostCheckpoint\n                import xgboost\n\n                train_X = np.array([[1, 2], [3, 4]])\n                train_y = np.array([0, 1])\n\n                model = xgboost.XGBClassifier().fit(train_X, train_y)\n                checkpoint = XGBoostCheckpoint.from_model(model.get_booster())\n\n        '
        tmpdir = tempfile.mkdtemp()
        booster.save_model(os.path.join(tmpdir, cls.MODEL_FILENAME))
        checkpoint = cls.from_directory(tmpdir)
        if preprocessor:
            checkpoint.set_preprocessor(preprocessor)
        return checkpoint

    def get_model(self) -> xgboost.Booster:
        if False:
            print('Hello World!')
        'Retrieve the XGBoost model stored in this checkpoint.'
        with self.as_directory() as checkpoint_path:
            booster = xgboost.Booster()
            booster.load_model(os.path.join(checkpoint_path, self.MODEL_FILENAME))
            return booster