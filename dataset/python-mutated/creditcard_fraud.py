import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class CreditCardFraudLoader(DatasetLoader):

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            while True:
                i = 10
        processed_df = super().transform_dataframe(dataframe)
        processed_df = processed_df.sort_values(by=['Time'])
        processed_df.loc[:198365, 'split'] = 0
        processed_df.loc[198365:, 'split'] = 2
        processed_df.split = processed_df.split.astype(int)
        return processed_df