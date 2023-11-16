import os
import tempfile
from typing import TYPE_CHECKING, Optional
import lightgbm
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI
if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

@PublicAPI(stability='beta')
class LightGBMCheckpoint(FrameworkCheckpoint):
    """A :py:class:`~ray.train.Checkpoint` with LightGBM-specific functionality."""
    MODEL_FILENAME = 'model.txt'

    @classmethod
    def from_model(cls, booster: lightgbm.Booster, *, preprocessor: Optional['Preprocessor']=None) -> 'LightGBMCheckpoint':
        if False:
            return 10
        'Create a :py:class:`~ray.train.Checkpoint` that stores a LightGBM model.\n\n        Args:\n            booster: The LightGBM model to store in the checkpoint.\n            preprocessor: A fitted preprocessor to be applied before inference.\n\n        Returns:\n            An :py:class:`LightGBMCheckpoint` containing the specified ``Estimator``.\n\n        Examples:\n            >>> import lightgbm\n            >>> import numpy as np\n            >>> from ray.train.lightgbm import LightGBMCheckpoint\n            >>>\n            >>> train_X = np.array([[1, 2], [3, 4]])\n            >>> train_y = np.array([0, 1])\n            >>>\n            >>> model = lightgbm.LGBMClassifier().fit(train_X, train_y)\n            >>> checkpoint = LightGBMCheckpoint.from_model(model.booster_)\n        '
        tempdir = tempfile.mkdtemp()
        booster.save_model(os.path.join(tempdir, cls.MODEL_FILENAME))
        checkpoint = cls.from_directory(tempdir)
        if preprocessor:
            checkpoint.set_preprocessor(preprocessor)
        return checkpoint

    def get_model(self) -> lightgbm.Booster:
        if False:
            print('Hello World!')
        'Retrieve the LightGBM model stored in this checkpoint.'
        with self.as_directory() as checkpoint_path:
            return lightgbm.Booster(model_file=os.path.join(checkpoint_path, self.MODEL_FILENAME))