from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.utils.module_loading import qualname
serializers = ['pandas.core.frame.DataFrame']
deserializers = serializers
if TYPE_CHECKING:
    import pandas as pd
    from airflow.serialization.serde import U
__version__ = 1

def serialize(o: object) -> tuple[U, str, int, bool]:
    if False:
        while True:
            i = 10
    import pandas as pd
    import pyarrow as pa
    from pyarrow import parquet as pq
    if not isinstance(o, pd.DataFrame):
        return ('', '', 0, False)
    table = pa.Table.from_pandas(o)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf, compression='snappy')
    return (buf.getvalue().hex().decode('utf-8'), qualname(o), __version__, True)

def deserialize(classname: str, version: int, data: object) -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    if version > __version__:
        raise TypeError(f'serialized {version} of {classname} > {__version__}')
    from pyarrow import parquet as pq
    if not isinstance(data, str):
        raise TypeError(f'serialized {classname} has wrong data type {type(data)}')
    from io import BytesIO
    with BytesIO(bytes.fromhex(data)) as buf:
        df = pq.read_table(buf).to_pandas()
    return df