import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class ConsumerComplaintsLoader(DatasetLoader):
    """The Consumer Complaints dataset."""

    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        'Loads a file into a dataframe.'
        consumer_complaints_df = pd.read_csv(file_path)
        consumer_complaints_df = preprocess_df(consumer_complaints_df)
        return consumer_complaints_df

def preprocess_df(df):
    if False:
        return 10
    'Preprocesses the dataframe.\n\n        - Remove all rows with missing values in the following columns:\n            - Consumer complaint narrative\n            - Issue\n            - Product\n\n    Args:\n        df (pd.DataFrame): The dataframe to preprocess.\n\n    Returns:\n        pd.DataFrame: The preprocessed dataframe.\n    '
    return df.dropna(subset=['Consumer complaint narrative', 'Issue', 'Product'])