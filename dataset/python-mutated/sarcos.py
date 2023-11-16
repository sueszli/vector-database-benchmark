import os
import pandas as pd
from scipy.io import loadmat
from ludwig.datasets.loaders.dataset_loader import DatasetLoader
from ludwig.utils.fs_utils import open_file

class SarcosLoader(DatasetLoader):
    """The Sarcos dataset.

    Details:
        The data relates to an inverse dynamics problem for a seven
        degrees-of-freedom SARCOS anthropomorphic robot arm. The
        task is to map from a 21-dimensional input space (7 joint
        positions, 7 joint velocities, 7 joint accelerations) to the
        corresponding 7 joint torques. There are 44,484 training
        examples and 4,449 test examples. The first 21 columns are
        the input variables, and the 22nd column is used as the target
        variable.

    Dataset source:
        Locally Weighted Projection RegressionL: An O(n) Algorithm for
        Incremental Real Time Learning in High Dimensional Space,
        S. Vijayakumar and S. Schaal, Proc ICML 2000.
        http://www.gaussianprocess.org/gpml/data/
    """

    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        if False:
            return 10
        'Loads a file into a dataframe.'
        with open_file(file_path) as f:
            mat = loadmat(f)
        file_df = pd.DataFrame(mat[os.path.basename(file_path).split('.')[0]])
        return file_df

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            print('Hello World!')
        processed_df = super().transform_dataframe(dataframe)
        columns = []
        columns += [f'position_{i}' for i in range(1, 8)]
        columns += [f'velocity_{i}' for i in range(1, 8)]
        columns += [f'acceleration_{i}' for i in range(1, 8)]
        columns += [f'torque_{i}' for i in range(1, 8)]
        columns += ['split']
        processed_df.columns = columns
        return processed_df