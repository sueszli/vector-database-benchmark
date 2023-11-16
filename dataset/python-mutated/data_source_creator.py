from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
from feast.data_source import DataSource
from feast.feature_logging import LoggingDestination
from feast.repo_config import FeastConfigBaseModel
from feast.saved_dataset import SavedDatasetStorage

class DataSourceCreator(ABC):

    def __init__(self, project_name: str, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.project_name = project_name

    @abstractmethod
    def create_data_source(self, df: pd.DataFrame, destination_name: str, event_timestamp_column='ts', created_timestamp_column='created_ts', field_mapping: Dict[str, str]=None, timestamp_field: Optional[str]=None) -> DataSource:
        if False:
            return 10
        '\n        Create a data source based on the dataframe. Implementing this method requires the underlying implementation to\n        persist the dataframe in offline store, using the destination string as a way to differentiate multiple\n        dataframes and data sources.\n\n        Args:\n            df: The dataframe to be used to create the data source.\n            destination_name: This str is used by the implementing classes to\n                isolate the multiple dataframes from each other.\n            event_timestamp_column: (Deprecated) Pass through for the underlying data source.\n            created_timestamp_column: Pass through for the underlying data source.\n            field_mapping: Pass through for the underlying data source.\n            timestamp_field: Pass through for the underlying data source.\n\n\n        Returns:\n            A Data source object, pointing to a table or file that is uploaded/persisted for the purpose of the\n            test.\n        '
        ...

    @abstractmethod
    def create_offline_store_config(self) -> FeastConfigBaseModel:
        if False:
            i = 10
            return i + 15
        ...

    @abstractmethod
    def create_saved_dataset_destination(self) -> SavedDatasetStorage:
        if False:
            while True:
                i = 10
        ...

    def create_logged_features_destination(self) -> LoggingDestination:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def teardown(self):
        if False:
            return 10
        ...