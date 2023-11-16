"""Provides file storage functionality from Google Cloud Storage."""
from __future__ import annotations
import functools
from google.cloud import storage
from typing import List, Optional, Union

@functools.lru_cache(maxsize=1)
def _get_client() -> storage.Client:
    if False:
        i = 10
        return i + 15
    'Gets Cloud Storage client.\n\n    Returns:\n        storage.Client. Cloud Storage client.\n    '
    return storage.Client()

@functools.lru_cache(maxsize=1)
def _get_bucket(bucket_name: str) -> storage.bucket.Bucket:
    if False:
        while True:
            i = 10
    'Gets Cloud Storage bucket.\n\n    Args:\n        bucket_name: str. The name of the storage bucket to return.\n\n    Returns:\n        storage.bucket.Bucket. Cloud Storage bucket.\n    '
    return _get_client().get_bucket(bucket_name)

def isfile(bucket_name: str, filepath: str) -> bool:
    if False:
        print('Hello World!')
    "Checks if the file with the given filepath exists in the GCS.\n\n    Args:\n        bucket_name: str. The name of the GCS bucket.\n        filepath: str. The path to the relevant file within the entity's\n            assets folder.\n\n    Returns:\n        bool. Whether the file exists in GCS.\n    "
    return _get_bucket(bucket_name).get_blob(filepath) is not None

def get(bucket_name: str, filepath: str) -> bytes:
    if False:
        i = 10
        return i + 15
    "Gets a file as an unencoded stream of raw bytes.\n\n    Args:\n        bucket_name: str. The name of the GCS bucket.\n        filepath: str. The path to the relevant file within the entity's\n            assets folder.\n\n    Returns:\n        bytes. Returns data a bytes.\n    "
    blob = _get_bucket(bucket_name).get_blob(filepath)
    data = blob.download_as_bytes()
    return data

def commit(bucket_name: str, filepath: str, raw_bytes: Union[bytes, str], mimetype: Optional[str]) -> None:
    if False:
        while True:
            i = 10
    "Commits raw_bytes to the relevant file in the entity's assets folder.\n\n    Args:\n        bucket_name: str. The name of the GCS bucket.\n        filepath: str. The path to the relevant file within the entity's\n            assets folder.\n        raw_bytes: str. The content to be stored in the file.\n        mimetype: str|None. The content-type of the cloud file.\n    "
    blob = _get_bucket(bucket_name).blob(filepath)
    blob.upload_from_string(raw_bytes, content_type=mimetype)

def delete(bucket_name: str, filepath: str) -> None:
    if False:
        print('Hello World!')
    "Deletes a file and the metadata associated with it.\n\n    Args:\n        bucket_name: str. The name of the GCS bucket.\n        filepath: str. The path to the relevant file within the entity's\n            assets folder.\n    "
    blob = _get_bucket(bucket_name).get_blob(filepath)
    blob.delete()

def copy(bucket_name: str, source_assets_path: str, dest_assets_path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Copies images from source_path.\n\n    Args:\n        bucket_name: str. The name of the GCS bucket.\n        source_assets_path: str. The path to the source entity's assets\n            folder.\n        dest_assets_path: str. The path to the relevant file within the entity's\n            assets folder.\n    "
    src_blob = _get_bucket(bucket_name).get_blob(source_assets_path)
    _get_bucket(bucket_name).copy_blob(src_blob, _get_bucket(bucket_name), new_name=dest_assets_path)

def listdir(bucket_name: str, dir_name: str) -> List[storage.blob.Blob]:
    if False:
        i = 10
        return i + 15
    "Lists all files in a directory.\n\n    Args:\n        bucket_name: str. The name of the GCS bucket.\n        dir_name: str. The directory whose files should be listed. This\n            should not start with '/'.\n\n    Returns:\n        list(Blob). A list of blobs.\n    "
    return list(_get_client().list_blobs(_get_bucket(bucket_name), prefix=dir_name))