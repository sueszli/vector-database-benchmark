import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class AdultCensusIncomeLoader(DatasetLoader):

    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        if False:
            return 10
        if file_path.endswith('.test'):
            return pd.read_csv(file_path, skiprows=1)
        return super().load_file_to_dataframe(file_path)

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            return 10
        processed_df = super().transform_dataframe(dataframe)
        processed_df['income'] = processed_df['income'].str.rstrip('.')
        processed_df['income'] = processed_df['income'].str.strip()
        return processed_df