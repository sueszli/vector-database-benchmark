import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class NavalLoader(DatasetLoader):

    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        if False:
            print('Hello World!')
        'Loads a file into a dataframe.'
        return pd.read_csv(file_path, header=None, sep='   ')