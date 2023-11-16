import logging
import traceback
import warnings
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from pyspark.sql import SparkSession
from feast import flags_helper
from feast.data_source import DataSource
from feast.errors import DataSourceNoNameException
from feast.infra.offline_stores.offline_utils import get_temp_entity_table_name
from feast.protos.feast.core.DataSource_pb2 import DataSource as DataSourceProto
from feast.protos.feast.core.SavedDataset_pb2 import SavedDatasetStorage as SavedDatasetStorageProto
from feast.repo_config import RepoConfig
from feast.saved_dataset import SavedDatasetStorage
from feast.type_map import spark_to_feast_value_type
from feast.value_type import ValueType
logger = logging.getLogger(__name__)

class SparkSourceFormat(Enum):
    csv = 'csv'
    json = 'json'
    parquet = 'parquet'
    delta = 'delta'
    avro = 'avro'

class SparkSource(DataSource):

    def __init__(self, *, name: Optional[str]=None, table: Optional[str]=None, query: Optional[str]=None, path: Optional[str]=None, file_format: Optional[str]=None, event_timestamp_column: Optional[str]=None, created_timestamp_column: Optional[str]=None, field_mapping: Optional[Dict[str, str]]=None, description: Optional[str]='', tags: Optional[Dict[str, str]]=None, owner: Optional[str]='', timestamp_field: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        if name is None and table is None:
            raise DataSourceNoNameException()
        name = name or table
        assert name
        super().__init__(name=name, timestamp_field=timestamp_field, created_timestamp_column=created_timestamp_column, field_mapping=field_mapping, description=description, tags=tags, owner=owner)
        if not flags_helper.is_test():
            warnings.warn('The spark data source API is an experimental feature in alpha development. This API is unstable and it could and most probably will be changed in the future.', RuntimeWarning)
        self.spark_options = SparkOptions(table=table, query=query, path=path, file_format=file_format)

    @property
    def table(self):
        if False:
            return 10
        '\n        Returns the table of this feature data source\n        '
        return self.spark_options.table

    @property
    def query(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the query of this feature data source\n        '
        return self.spark_options.query

    @property
    def path(self):
        if False:
            print('Hello World!')
        '\n        Returns the path of the spark data source file.\n        '
        return self.spark_options.path

    @property
    def file_format(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the file format of this feature data source.\n        '
        return self.spark_options.file_format

    @staticmethod
    def from_proto(data_source: DataSourceProto) -> Any:
        if False:
            return 10
        assert data_source.HasField('spark_options')
        spark_options = SparkOptions.from_proto(data_source.spark_options)
        return SparkSource(name=data_source.name, field_mapping=dict(data_source.field_mapping), table=spark_options.table, query=spark_options.query, path=spark_options.path, file_format=spark_options.file_format, timestamp_field=data_source.timestamp_field, created_timestamp_column=data_source.created_timestamp_column, description=data_source.description, tags=dict(data_source.tags), owner=data_source.owner)

    def to_proto(self) -> DataSourceProto:
        if False:
            while True:
                i = 10
        data_source_proto = DataSourceProto(name=self.name, type=DataSourceProto.BATCH_SPARK, data_source_class_type='feast.infra.offline_stores.contrib.spark_offline_store.spark_source.SparkSource', field_mapping=self.field_mapping, spark_options=self.spark_options.to_proto(), description=self.description, tags=self.tags, owner=self.owner)
        data_source_proto.timestamp_field = self.timestamp_field
        data_source_proto.created_timestamp_column = self.created_timestamp_column
        return data_source_proto

    def validate(self, config: RepoConfig):
        if False:
            return 10
        self.get_table_column_names_and_types(config)

    @staticmethod
    def source_datatype_to_feast_value_type() -> Callable[[str], ValueType]:
        if False:
            i = 10
            return i + 15
        return spark_to_feast_value_type

    def get_table_column_names_and_types(self, config: RepoConfig) -> Iterable[Tuple[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        from feast.infra.offline_stores.contrib.spark_offline_store.spark import get_spark_session_or_start_new_with_repoconfig
        spark_session = get_spark_session_or_start_new_with_repoconfig(store_config=config.offline_store)
        df = spark_session.sql(f'SELECT * FROM {self.get_table_query_string()}')
        return ((field.name, field.dataType.simpleString()) for field in df.schema)

    def get_table_query_string(self) -> str:
        if False:
            print('Hello World!')
        'Returns a string that can directly be used to reference this table in SQL'
        if self.table:
            table = '.'.join([f'`{x}`' for x in self.table.split('.')])
            return table
        if self.query:
            return f'({self.query})'
        spark_session = SparkSession.getActiveSession()
        if spark_session is None:
            raise AssertionError('Could not find an active spark session.')
        try:
            df = spark_session.read.format(self.file_format).load(self.path)
        except Exception:
            logger.exception('Spark read of file source failed.\n' + traceback.format_exc())
        tmp_table_name = get_temp_entity_table_name()
        df.createOrReplaceTempView(tmp_table_name)
        return f'`{tmp_table_name}`'

class SparkOptions:
    allowed_formats = [format.value for format in SparkSourceFormat]

    def __init__(self, table: Optional[str], query: Optional[str], path: Optional[str], file_format: Optional[str]):
        if False:
            for i in range(10):
                print('nop')
        if sum([not not arg for arg in [table, query, path]]) != 1:
            raise ValueError('Exactly one of params(table, query, path) must be specified.')
        if path:
            if not file_format:
                raise ValueError("If 'path' is specified, then 'file_format' is required.")
            if file_format not in self.allowed_formats:
                raise ValueError(f"'file_format' should be one of {self.allowed_formats}")
        self._table = table
        self._query = query
        self._path = path
        self._file_format = file_format

    @property
    def table(self):
        if False:
            for i in range(10):
                print('nop')
        return self._table

    @table.setter
    def table(self, table):
        if False:
            return 10
        self._table = table

    @property
    def query(self):
        if False:
            i = 10
            return i + 15
        return self._query

    @query.setter
    def query(self, query):
        if False:
            print('Hello World!')
        self._query = query

    @property
    def path(self):
        if False:
            print('Hello World!')
        return self._path

    @path.setter
    def path(self, path):
        if False:
            i = 10
            return i + 15
        self._path = path

    @property
    def file_format(self):
        if False:
            print('Hello World!')
        return self._file_format

    @file_format.setter
    def file_format(self, file_format):
        if False:
            for i in range(10):
                print('nop')
        self._file_format = file_format

    @classmethod
    def from_proto(cls, spark_options_proto: DataSourceProto.SparkOptions):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a SparkOptions from a protobuf representation of a spark option\n        args:\n            spark_options_proto: a protobuf representation of a datasource\n        Returns:\n            Returns a SparkOptions object based on the spark_options protobuf\n        '
        spark_options = cls(table=spark_options_proto.table, query=spark_options_proto.query, path=spark_options_proto.path, file_format=spark_options_proto.file_format)
        return spark_options

    def to_proto(self) -> DataSourceProto.SparkOptions:
        if False:
            i = 10
            return i + 15
        '\n        Converts an SparkOptionsProto object to its protobuf representation.\n        Returns:\n            SparkOptionsProto protobuf\n        '
        spark_options_proto = DataSourceProto.SparkOptions(table=self.table, query=self.query, path=self.path, file_format=self.file_format)
        return spark_options_proto

class SavedDatasetSparkStorage(SavedDatasetStorage):
    _proto_attr_name = 'spark_storage'
    spark_options: SparkOptions

    def __init__(self, table: Optional[str]=None, query: Optional[str]=None, path: Optional[str]=None, file_format: Optional[str]=None):
        if False:
            print('Hello World!')
        self.spark_options = SparkOptions(table=table, query=query, path=path, file_format=file_format)

    @staticmethod
    def from_proto(storage_proto: SavedDatasetStorageProto) -> SavedDatasetStorage:
        if False:
            for i in range(10):
                print('nop')
        spark_options = SparkOptions.from_proto(storage_proto.spark_storage)
        return SavedDatasetSparkStorage(table=spark_options.table, query=spark_options.query, path=spark_options.path, file_format=spark_options.file_format)

    def to_proto(self) -> SavedDatasetStorageProto:
        if False:
            print('Hello World!')
        return SavedDatasetStorageProto(spark_storage=self.spark_options.to_proto())

    def to_data_source(self) -> DataSource:
        if False:
            return 10
        return SparkSource(table=self.spark_options.table, query=self.spark_options.query, path=self.spark_options.path, file_format=self.spark_options.file_format)