import numpy as np
import pandas as pd
from ludwig.constants import SPLIT
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class RandomSplitLoader(DatasetLoader):
    """Adds a random split column to the dataset, with fixed proportions of:
     train: 70%
     validation: 10%
     test: 20%
    ."""

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        df = super().transform_dataframe(dataframe)
        df[SPLIT] = np.random.choice(3, len(df), p=(0.7, 0.1, 0.2)).astype(np.int8)
        return df