import os
import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class AllstateClaimsSeverityLoader(DatasetLoader):

    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        if os.path.basename(file_path) == 'train.csv':
            return pd.read_csv(file_path, nrows=188319)
        if os.path.basename(file_path) == 'test.csv':
            return pd.read_csv(file_path, nrows=125547)
        super().load_file_to_dataframe(file_path)