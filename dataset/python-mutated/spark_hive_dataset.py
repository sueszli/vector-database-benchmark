"""``AbstractDataset`` implementation to access Spark dataframes using
``pyspark`` on Apache Hive.
"""
import pickle
from copy import deepcopy
from typing import Any, Dict, List
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import col, lit, row_number
from kedro.io.core import AbstractDataset, DatasetError

class SparkHiveDataSet(AbstractDataset[DataFrame, DataFrame]):
    """``SparkHiveDataSet`` loads and saves Spark dataframes stored on Hive.
    This data set also handles some incompatible file types such as using partitioned parquet on
    hive which will not normally allow upserts to existing data without a complete replacement
    of the existing file/partition.

    This DataSet has some key assumptions:

    - Schemas do not change during the pipeline run (defined PKs must be present for the
      duration of the pipeline)
    - Tables are not being externally modified during upserts. The upsert method is NOT ATOMIC

    to external changes to the target table while executing.
    Upsert methodology works by leveraging Spark DataFrame execution plan checkpointing.

    Example usage for the
    `YAML API <https://kedro.readthedocs.io/en/stable/data/    data_catalog_yaml_examples.html>`_:


    .. code-block:: yaml

        hive_dataset:
          type: spark.SparkHiveDataSet
          database: hive_database
          table: table_name
          write_mode: overwrite

    Example usage for the
    `Python API <https://kedro.readthedocs.io/en/stable/data/    advanced_data_catalog_usage.html>`_:
    ::

        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql.types import (StructField, StringType,
        >>>                                IntegerType, StructType)
        >>>
        >>> from kedro.extras.datasets.spark import SparkHiveDataSet
        >>>
        >>> schema = StructType([StructField("name", StringType(), True),
        >>>                      StructField("age", IntegerType(), True)])
        >>>
        >>> data = [('Alex', 31), ('Bob', 12), ('Clarke', 65), ('Dave', 29)]
        >>>
        >>> spark_df = SparkSession.builder.getOrCreate().createDataFrame(data, schema)
        >>>
        >>> data_set = SparkHiveDataSet(database="test_database", table="test_table",
        >>>                             write_mode="overwrite")
        >>> data_set.save(spark_df)
        >>> reloaded = data_set.load()
        >>>
        >>> reloaded.take(4)
    """
    DEFAULT_SAVE_ARGS = {}

    def __init__(self, database: str, table: str, write_mode: str='errorifexists', table_pk: List[str]=None, save_args: Dict[str, Any]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Creates a new instance of ``SparkHiveDataSet``.\n\n        Args:\n            database: The name of the hive database.\n            table: The name of the table within the database.\n            write_mode: ``insert``, ``upsert`` or ``overwrite`` are supported.\n            table_pk: If performing an upsert, this identifies the primary key columns used to\n                resolve preexisting data. Is required for ``write_mode="upsert"``.\n            save_args: Optional mapping of any options,\n                passed to the `DataFrameWriter.saveAsTable` as kwargs.\n                Key example of this is `partitionBy` which allows data partitioning\n                on a list of column names.\n                Other `HiveOptions` can be found here:\n                https://spark.apache.org/docs/latest/sql-data-sources-hive-tables.html#specifying-storage-format-for-hive-tables\n\n        Note:\n            For users leveraging the `upsert` functionality,\n            a `checkpoint` directory must be set, e.g. using\n            `spark.sparkContext.setCheckpointDir("/path/to/dir")`\n            or directly in the Spark conf folder.\n\n        Raises:\n            DatasetError: Invalid configuration supplied\n        '
        _write_modes = ['append', 'error', 'errorifexists', 'upsert', 'overwrite']
        if write_mode not in _write_modes:
            valid_modes = ', '.join(_write_modes)
            raise DatasetError(f"Invalid 'write_mode' provided: {write_mode}. 'write_mode' must be one of: {valid_modes}")
        if write_mode == 'upsert' and (not table_pk):
            raise DatasetError("'table_pk' must be set to utilise 'upsert' read mode")
        self._write_mode = write_mode
        self._table_pk = table_pk or []
        self._database = database
        self._table = table
        self._full_table_address = f'{database}.{table}'
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)
        self._format = self._save_args.pop('format', None) or 'hive'
        self._eager_checkpoint = self._save_args.pop('eager_checkpoint', None) or True

    def _describe(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {'database': self._database, 'table': self._table, 'write_mode': self._write_mode, 'table_pk': self._table_pk, 'partition_by': self._save_args.get('partitionBy'), 'format': self._format}

    @staticmethod
    def _get_spark() -> SparkSession:
        if False:
            return 10
        '\n        This method should only be used to get an existing SparkSession\n        with valid Hive configuration.\n        Configuration for Hive is read from hive-site.xml on the classpath.\n        It supports running both SQL and HiveQL commands.\n        Additionally, if users are leveraging the `upsert` functionality,\n        then a `checkpoint` directory must be set, e.g. using\n        `spark.sparkContext.setCheckpointDir("/path/to/dir")`\n        '
        _spark = SparkSession.builder.getOrCreate()
        return _spark

    def _create_hive_table(self, data: DataFrame, mode: str=None):
        if False:
            i = 10
            return i + 15
        _mode: str = mode or self._write_mode
        data.write.saveAsTable(self._full_table_address, mode=_mode, format=self._format, **self._save_args)

    def _load(self) -> DataFrame:
        if False:
            while True:
                i = 10
        return self._get_spark().read.table(self._full_table_address)

    def _save(self, data: DataFrame) -> None:
        if False:
            print('Hello World!')
        self._validate_save(data)
        if self._write_mode == 'upsert':
            if not set(self._table_pk) <= set(self._load().columns):
                raise DatasetError(f'Columns {str(self._table_pk)} selected as primary key(s) not found in table {self._full_table_address}')
            self._upsert_save(data=data)
        else:
            self._create_hive_table(data=data)

    def _upsert_save(self, data: DataFrame) -> None:
        if False:
            print('Hello World!')
        if not self._exists() or self._load().rdd.isEmpty():
            self._create_hive_table(data=data, mode='overwrite')
        else:
            _tmp_colname = 'tmp_colname'
            _tmp_row = 'tmp_row'
            _w = Window.partitionBy(*self._table_pk).orderBy(col(_tmp_colname).desc())
            df_old = self._load().select('*', lit(1).alias(_tmp_colname))
            df_new = data.select('*', lit(2).alias(_tmp_colname))
            df_stacked = df_new.unionByName(df_old).select('*', row_number().over(_w).alias(_tmp_row))
            df_filtered = df_stacked.filter(col(_tmp_row) == 1).drop(_tmp_colname, _tmp_row).checkpoint(eager=self._eager_checkpoint)
            self._create_hive_table(data=df_filtered, mode='overwrite')

    def _validate_save(self, data: DataFrame):
        if False:
            while True:
                i = 10
        if not self._exists() or self._write_mode == 'overwrite':
            return
        hive_dtypes = set(self._load().dtypes)
        data_dtypes = set(data.dtypes)
        if data_dtypes != hive_dtypes:
            new_cols = data_dtypes - hive_dtypes
            missing_cols = hive_dtypes - data_dtypes
            raise DatasetError(f'Dataset does not match hive table schema.\nPresent on insert only: {sorted(new_cols)}\nPresent on schema only: {sorted(missing_cols)}')

    def _exists(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._get_spark()._jsparkSession.catalog().tableExists(self._database, self._table)

    def __getstate__(self) -> None:
        if False:
            while True:
                i = 10
        raise pickle.PicklingError('PySpark datasets objects cannot be pickled or serialised as Python objects.')