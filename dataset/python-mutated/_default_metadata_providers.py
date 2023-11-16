from typing import List, Optional
from ray.data.datasource import DefaultFileMetadataProvider, DefaultParquetMetadataProvider, FastFileMetadataProvider
from ray.data.datasource.image_datasource import _ImageFileMetadataProvider

def get_generic_metadata_provider(file_extensions: Optional[List[str]]):
    if False:
        return 10
    return DefaultFileMetadataProvider()

def get_parquet_metadata_provider():
    if False:
        i = 10
        return i + 15
    return DefaultParquetMetadataProvider()

def get_parquet_bulk_metadata_provider():
    if False:
        for i in range(10):
            print('nop')
    return FastFileMetadataProvider()

def get_image_metadata_provider():
    if False:
        return 10
    return _ImageFileMetadataProvider()