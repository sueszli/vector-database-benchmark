import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class SantanderValuePredictionLoader(DatasetLoader):
    """The Santander Value Prediction Challenge dataset.

    https://www.kaggle.com/c/santander-value-prediction-challenge
    """

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            return 10
        processed_df = super().transform_dataframe(dataframe)
        processed_df.columns = ['C' + str(col) for col in processed_df.columns]
        processed_df.rename(columns={'CID': 'ID', 'Ctarget': 'target', 'Csplit': 'split'}, inplace=True)
        return processed_df