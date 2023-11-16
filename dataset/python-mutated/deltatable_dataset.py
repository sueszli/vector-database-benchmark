"""``AbstractDataset`` implementation to access DeltaTables using
``delta-spark``
"""
from pathlib import PurePosixPath
from typing import NoReturn
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from kedro.extras.datasets.spark.spark_dataset import _split_filepath, _strip_dbfs_prefix
from kedro.io.core import AbstractDataset, DatasetError

class DeltaTableDataSet(AbstractDataset[None, DeltaTable]):
    """``DeltaTableDataSet`` loads data into DeltaTable objects.

    Example usage for the
    `YAML API <https://kedro.readthedocs.io/en/stable/data/    data_catalog_yaml_examples.html>`_:


    .. code-block:: yaml

        weather@spark:
          type: spark.SparkDataSet
          filepath: data/02_intermediate/data.parquet
          file_format: "delta"

        weather@delta:
          type: spark.DeltaTableDataSet
          filepath: data/02_intermediate/data.parquet

    Example usage for the
    `Python API <https://kedro.readthedocs.io/en/stable/data/    advanced_data_catalog_usage.html>`_:
    ::

        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql.types import (StructField, StringType,
        >>>                                IntegerType, StructType)
        >>>
        >>> from kedro.extras.datasets.spark import DeltaTableDataSet, SparkDataSet
        >>>
        >>> schema = StructType([StructField("name", StringType(), True),
        >>>                      StructField("age", IntegerType(), True)])
        >>>
        >>> data = [('Alex', 31), ('Bob', 12), ('Clarke', 65), ('Dave', 29)]
        >>>
        >>> spark_df = SparkSession.builder.getOrCreate().createDataFrame(data, schema)
        >>>
        >>> data_set = SparkDataSet(filepath="test_data", file_format="delta")
        >>> data_set.save(spark_df)
        >>> deltatable_dataset = DeltaTableDataSet(filepath="test_data")
        >>> delta_table = deltatable_dataset.load()
        >>>
        >>> delta_table.update()
    """
    _SINGLE_PROCESS = True

    def __init__(self, filepath: str) -> None:
        if False:
            print('Hello World!')
        'Creates a new instance of ``DeltaTableDataSet``.\n\n        Args:\n            filepath: Filepath in POSIX format to a Spark dataframe. When using Databricks\n                and working with data written to mount path points,\n                specify ``filepath``s for (versioned) ``SparkDataSet``s\n                starting with ``/dbfs/mnt``.\n        '
        (fs_prefix, filepath) = _split_filepath(filepath)
        self._fs_prefix = fs_prefix
        self._filepath = PurePosixPath(filepath)

    @staticmethod
    def _get_spark():
        if False:
            for i in range(10):
                print('nop')
        return SparkSession.builder.getOrCreate()

    def _load(self) -> DeltaTable:
        if False:
            for i in range(10):
                print('nop')
        load_path = self._fs_prefix + str(self._filepath)
        return DeltaTable.forPath(self._get_spark(), load_path)

    def _save(self, data: None) -> NoReturn:
        if False:
            print('Hello World!')
        raise DatasetError(f'{self.__class__.__name__} is a read only dataset type')

    def _exists(self) -> bool:
        if False:
            i = 10
            return i + 15
        load_path = _strip_dbfs_prefix(self._fs_prefix + str(self._filepath))
        try:
            self._get_spark().read.load(path=load_path, format='delta')
        except AnalysisException as exception:
            message = exception.desc if hasattr(exception, 'desc') else exception.message
            if 'Path does not exist:' in message or 'is not a Delta table' in message:
                return False
            raise
        return True

    def _describe(self):
        if False:
            for i in range(10):
                print('nop')
        return {'filepath': str(self._filepath), 'fs_prefix': self._fs_prefix}