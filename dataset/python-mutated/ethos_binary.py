import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class EthosBinaryLoader(DatasetLoader):

    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        if False:
            while True:
                i = 10
        return pd.read_csv(file_path, sep=';')

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        processed_df = super().transform_dataframe(dataframe)
        processed_df['isHate'] = processed_df['isHate'] >= 0.5
        processed_df['isHate'] = processed_df['isHate'].astype(int)
        return processed_df