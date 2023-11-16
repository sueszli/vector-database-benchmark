import os
import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class InsuranceLiteLoader(DatasetLoader):
    """Health Insurance Cross Sell Prediction Predict Health Insurance Owners' who will be interested in Vehicle
    Insurance https://www.kaggle.com/datasets/arashnic/imbalanced-data-practice."""

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        df = super().transform_dataframe(dataframe)
        df['image_path'] = df['image_path'].apply(lambda x: os.path.join('Fast_Furious_Insured', 'trainImages', os.path.basename(x)))
        return df