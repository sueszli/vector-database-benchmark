from typing import Any, Dict, Optional, Sequence, Tuple, Type
import polars as pl
import pyarrow as pa
from dagster._core.storage.db_io_manager import DbTypeHandler
from dagster_deltalake.handler import DeltalakeBaseArrowTypeHandler, DeltaLakePyArrowTypeHandler
from dagster_deltalake.io_manager import DeltaLakeIOManager

class DeltaLakePolarsTypeHandler(DeltalakeBaseArrowTypeHandler[pl.DataFrame]):

    def from_arrow(self, obj: pa.RecordBatchReader, target_type: Type[pl.DataFrame]) -> pl.DataFrame:
        if False:
            i = 10
            return i + 15
        return pl.from_arrow(obj)

    def to_arrow(self, obj: pl.DataFrame) -> Tuple[pa.RecordBatchReader, Dict[str, Any]]:
        if False:
            while True:
                i = 10
        return (obj.to_arrow().to_reader(), {'large_dtypes': True})

    @property
    def supported_types(self) -> Sequence[Type[object]]:
        if False:
            for i in range(10):
                print('nop')
        return [pl.DataFrame]

class DeltaLakePolarsIOManager(DeltaLakeIOManager):

    @staticmethod
    def type_handlers() -> Sequence[DbTypeHandler]:
        if False:
            return 10
        return [DeltaLakePolarsTypeHandler(), DeltaLakePyArrowTypeHandler()]

    @staticmethod
    def default_load_type() -> Optional[Type]:
        if False:
            print('Hello World!')
        return pl.DataFrame