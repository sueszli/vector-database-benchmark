import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
import pandas as pd
import pyarrow
from feast import flags_helper
from feast.data_source import DataSource
from feast.dqm.errors import ValidationFailed
from feast.feature_logging import LoggingConfig, LoggingSource
from feast.feature_view import FeatureView
from feast.infra.registry.base_registry import BaseRegistry
from feast.on_demand_feature_view import OnDemandFeatureView
from feast.repo_config import RepoConfig
from feast.saved_dataset import SavedDatasetStorage
if TYPE_CHECKING:
    from feast.saved_dataset import ValidationReference
warnings.simplefilter('once', RuntimeWarning)

class RetrievalMetadata:
    min_event_timestamp: Optional[datetime]
    max_event_timestamp: Optional[datetime]
    features: List[str]
    keys: List[str]

    def __init__(self, features: List[str], keys: List[str], min_event_timestamp: Optional[datetime]=None, max_event_timestamp: Optional[datetime]=None):
        if False:
            i = 10
            return i + 15
        self.features = features
        self.keys = keys
        self.min_event_timestamp = min_event_timestamp
        self.max_event_timestamp = max_event_timestamp

class RetrievalJob(ABC):
    """A RetrievalJob manages the execution of a query to retrieve data from the offline store."""

    def to_df(self, validation_reference: Optional['ValidationReference']=None, timeout: Optional[int]=None) -> pd.DataFrame:
        if False:
            while True:
                i = 10
        '\n        Synchronously executes the underlying query and returns the result as a pandas dataframe.\n\n        On demand transformations will be executed. If a validation reference is provided, the dataframe\n        will be validated.\n\n        Args:\n            validation_reference (optional): The validation to apply against the retrieved dataframe.\n            timeout (optional): The query timeout if applicable.\n        '
        features_df = self._to_df_internal(timeout=timeout)
        if self.on_demand_feature_views:
            for odfv in self.on_demand_feature_views:
                features_df = features_df.join(odfv.get_transformed_features_df(features_df, self.full_feature_names))
        if validation_reference:
            if not flags_helper.is_test():
                warnings.warn('Dataset validation is an experimental feature. This API is unstable and it could and most probably will be changed in the future. We do not guarantee that future changes will maintain backward compatibility.', RuntimeWarning)
            validation_result = validation_reference.profile.validate(features_df)
            if not validation_result.is_success:
                raise ValidationFailed(validation_result)
        return features_df

    def to_arrow(self, validation_reference: Optional['ValidationReference']=None, timeout: Optional[int]=None) -> pyarrow.Table:
        if False:
            for i in range(10):
                print('nop')
        '\n        Synchronously executes the underlying query and returns the result as an arrow table.\n\n        On demand transformations will be executed. If a validation reference is provided, the dataframe\n        will be validated.\n\n        Args:\n            validation_reference (optional): The validation to apply against the retrieved dataframe.\n            timeout (optional): The query timeout if applicable.\n        '
        if not self.on_demand_feature_views and (not validation_reference):
            return self._to_arrow_internal(timeout=timeout)
        features_df = self._to_df_internal(timeout=timeout)
        if self.on_demand_feature_views:
            for odfv in self.on_demand_feature_views:
                features_df = features_df.join(odfv.get_transformed_features_df(features_df, self.full_feature_names))
        if validation_reference:
            if not flags_helper.is_test():
                warnings.warn('Dataset validation is an experimental feature. This API is unstable and it could and most probably will be changed in the future. We do not guarantee that future changes will maintain backward compatibility.', RuntimeWarning)
            validation_result = validation_reference.profile.validate(features_df)
            if not validation_result.is_success:
                raise ValidationFailed(validation_result)
        return pyarrow.Table.from_pandas(features_df)

    def to_sql(self) -> str:
        if False:
            print('Hello World!')
        '\n        Return RetrievalJob generated SQL statement if applicable.\n        '
        pass

    @abstractmethod
    def _to_df_internal(self, timeout: Optional[int]=None) -> pd.DataFrame:
        if False:
            print('Hello World!')
        '\n        Synchronously executes the underlying query and returns the result as a pandas dataframe.\n\n        timeout: RetreivalJob implementations may implement a timeout.\n\n        Does not handle on demand transformations or dataset validation. For either of those,\n        `to_df` should be used.\n        '
        pass

    @abstractmethod
    def _to_arrow_internal(self, timeout: Optional[int]=None) -> pyarrow.Table:
        if False:
            print('Hello World!')
        '\n        Synchronously executes the underlying query and returns the result as an arrow table.\n\n        timeout: RetreivalJob implementations may implement a timeout.\n\n        Does not handle on demand transformations or dataset validation. For either of those,\n        `to_arrow` should be used.\n        '
        pass

    @property
    @abstractmethod
    def full_feature_names(self) -> bool:
        if False:
            return 10
        'Returns True if full feature names should be applied to the results of the query.'
        pass

    @property
    @abstractmethod
    def on_demand_feature_views(self) -> List[OnDemandFeatureView]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a list containing all the on demand feature views to be handled.'
        pass

    @abstractmethod
    def persist(self, storage: SavedDatasetStorage, allow_overwrite: bool=False, timeout: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Synchronously executes the underlying query and persists the result in the same offline store\n        at the specified destination.\n\n        Args:\n            storage: The saved dataset storage object specifying where the result should be persisted.\n            allow_overwrite: If True, a pre-existing location (e.g. table or file) can be overwritten.\n                Currently not all individual offline store implementations make use of this parameter.\n        '
        pass

    @property
    @abstractmethod
    def metadata(self) -> Optional[RetrievalMetadata]:
        if False:
            i = 10
            return i + 15
        'Returns metadata about the retrieval job.'
        pass

    def supports_remote_storage_export(self) -> bool:
        if False:
            return 10
        'Returns True if the RetrievalJob supports `to_remote_storage`.'
        return False

    def to_remote_storage(self) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Synchronously executes the underlying query and exports the results to remote storage (e.g. S3 or GCS).\n\n        Implementations of this method should export the results as multiple parquet files, each file sized\n        appropriately depending on how much data is being returned by the retrieval job.\n\n        Returns:\n            A list of parquet file paths in remote storage.\n        '
        raise NotImplementedError()

class OfflineStore(ABC):
    """
    An offline store defines the interface that Feast uses to interact with the storage and compute system that
    handles offline features.

    Each offline store implementation is designed to work only with the corresponding data source. For example,
    the SnowflakeOfflineStore can handle SnowflakeSources but not FileSources.
    """

    @staticmethod
    @abstractmethod
    def pull_latest_from_table_or_query(config: RepoConfig, data_source: DataSource, join_key_columns: List[str], feature_name_columns: List[str], timestamp_field: str, created_timestamp_column: Optional[str], start_date: datetime, end_date: datetime) -> RetrievalJob:
        if False:
            print('Hello World!')
        '\n        Extracts the latest entity rows (i.e. the combination of join key columns, feature columns, and\n        timestamp columns) from the specified data source that lie within the specified time range.\n\n        All of the column names should refer to columns that exist in the data source. In particular,\n        any mapping of column names must have already happened.\n\n        Args:\n            config: The config for the current feature store.\n            data_source: The data source from which the entity rows will be extracted.\n            join_key_columns: The columns of the join keys.\n            feature_name_columns: The columns of the features.\n            timestamp_field: The timestamp column, used to determine which rows are the most recent.\n            created_timestamp_column: The column indicating when the row was created, used to break ties.\n            start_date: The start of the time range.\n            end_date: The end of the time range.\n\n        Returns:\n            A RetrievalJob that can be executed to get the entity rows.\n        '
        pass

    @staticmethod
    @abstractmethod
    def get_historical_features(config: RepoConfig, feature_views: List[FeatureView], feature_refs: List[str], entity_df: Union[pd.DataFrame, str], registry: BaseRegistry, project: str, full_feature_names: bool=False) -> RetrievalJob:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieves the point-in-time correct historical feature values for the specified entity rows.\n\n        Args:\n            config: The config for the current feature store.\n            feature_views: A list containing all feature views that are referenced in the entity rows.\n            feature_refs: The features to be retrieved.\n            entity_df: A collection of rows containing all entity columns on which features need to be joined,\n                as well as the timestamp column used for point-in-time joins. Either a pandas dataframe can be\n                provided or a SQL query.\n            registry: The registry for the current feature store.\n            project: Feast project to which the feature views belong.\n            full_feature_names: If True, feature names will be prefixed with the corresponding feature view name,\n                changing them from the format "feature" to "feature_view__feature" (e.g. "daily_transactions"\n                changes to "customer_fv__daily_transactions").\n\n        Returns:\n            A RetrievalJob that can be executed to get the features.\n        '
        pass

    @staticmethod
    @abstractmethod
    def pull_all_from_table_or_query(config: RepoConfig, data_source: DataSource, join_key_columns: List[str], feature_name_columns: List[str], timestamp_field: str, start_date: datetime, end_date: datetime) -> RetrievalJob:
        if False:
            while True:
                i = 10
        '\n        Extracts all the entity rows (i.e. the combination of join key columns, feature columns, and\n        timestamp columns) from the specified data source that lie within the specified time range.\n\n        All of the column names should refer to columns that exist in the data source. In particular,\n        any mapping of column names must have already happened.\n\n        Args:\n            config: The config for the current feature store.\n            data_source: The data source from which the entity rows will be extracted.\n            join_key_columns: The columns of the join keys.\n            feature_name_columns: The columns of the features.\n            timestamp_field: The timestamp column.\n            start_date: The start of the time range.\n            end_date: The end of the time range.\n\n        Returns:\n            A RetrievalJob that can be executed to get the entity rows.\n        '
        pass

    @staticmethod
    def write_logged_features(config: RepoConfig, data: Union[pyarrow.Table, Path], source: LoggingSource, logging_config: LoggingConfig, registry: BaseRegistry):
        if False:
            i = 10
            return i + 15
        '\n        Writes logged features to a specified destination in the offline store.\n\n        If the specified destination exists, data will be appended; otherwise, the destination will be\n        created and data will be added. Thus this function can be called repeatedly with the same\n        destination to flush logs in chunks.\n\n        Args:\n            config: The config for the current feature store.\n            data: An arrow table or a path to parquet directory that contains the logs to write.\n            source: The logging source that provides a schema and some additional metadata.\n            logging_config: A LoggingConfig object that determines where the logs will be written.\n            registry: The registry for the current feature store.\n        '
        raise NotImplementedError()

    @staticmethod
    def offline_write_batch(config: RepoConfig, feature_view: FeatureView, table: pyarrow.Table, progress: Optional[Callable[[int], Any]]):
        if False:
            return 10
        '\n        Writes the specified arrow table to the data source underlying the specified feature view.\n\n        Args:\n            config: The config for the current feature store.\n            feature_view: The feature view whose batch source should be written.\n            table: The arrow table to write.\n            progress: Function to be called once a portion of the data has been written, used\n                to show progress.\n        '
        raise NotImplementedError()