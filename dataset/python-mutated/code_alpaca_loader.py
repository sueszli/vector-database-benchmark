import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class CodeAlpacaLoader(DatasetLoader):
    """The Code Alpaca dataset."""

    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        'Loads a file into a dataframe.'
        df = pd.read_json(file_path)
        return df