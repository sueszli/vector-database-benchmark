import os
from typing import List
import pandas as pd
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class IEEEFraudLoader(DatasetLoader):
    """The IEEE-CIS Fraud Detection Dataset https://www.kaggle.com/c/ieee-fraud-detection/overview."""

    def load_unprocessed_dataframe(self, file_paths: List[str]) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        'Load dataset files into a dataframe.'
        train_files = {'train_identity.csv', 'train_transaction.csv'}
        test_files = {'test_identity.csv', 'test_transaction.csv'}
        (train_dfs, test_dfs) = ({}, {})
        for filename in train_files.union(test_files):
            split_name = os.path.splitext(filename)[0]
            file_df = self.load_file_to_dataframe(os.path.join(self.raw_dataset_dir, filename))
            if filename in train_files:
                train_dfs[split_name] = file_df
            elif filename in test_files:
                test_dfs[split_name] = file_df
        final_train = pd.merge(train_dfs['train_transaction'], train_dfs['train_identity'], on='TransactionID', how='left')
        return final_train