from __future__ import annotations
from pathlib import Path
from typing import Callable, Iterable, TypeVar
import pyarrow as pa
import pyarrow.parquet as pq

def stream_to_parquet(path: Path, tables: Iterable[pa.Table]) -> None:
    if False:
        print('Hello World!')
    try:
        first = next(tables)
    except StopIteration:
        return
    schema = first.schema
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_table(first)
        for table in tables:
            table = table.cast(schema)
            writer.write_table(table)

def stream_from_parquet(path: Path) -> Iterable[pa.Table]:
    if False:
        for i in range(10):
            print('nop')
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches():
        yield pa.Table.from_batches([batch])

def stream_from_parquets(paths: Iterable[Path]) -> Iterable[pa.Table]:
    if False:
        for i in range(10):
            print('nop')
    for path in paths:
        yield from stream_from_parquet(path)
T = TypeVar('T')

def coalesce(items: Iterable[T], max_size: int, sizer: Callable[[T], int]=len) -> Iterable[list[T]]:
    if False:
        print('Hello World!')
    batch = []
    current_size = 0
    for item in items:
        this_size = sizer(item)
        if current_size + this_size > max_size:
            yield batch
            batch = []
            current_size = 0
        batch.append(item)
        current_size += this_size
    if batch:
        yield batch

def coalesce_parquets(paths: Iterable[Path], outpath: Path, max_size: int=2 ** 20) -> None:
    if False:
        return 10
    tables = stream_from_parquets(paths)
    table_groups = coalesce(tables, max_size)
    coalesced_tables = (pa.concat_tables(group) for group in table_groups)
    stream_to_parquet(outpath, coalesced_tables)

def merge_parquet_dir(path: str, outpath: Path) -> None:
    if False:
        for i in range(10):
            print('nop')
    paths = Path(path).glob('*.parquet')
    coalesce_parquets(paths, outpath)