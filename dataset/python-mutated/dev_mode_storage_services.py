"""Provides various functions from the Cloud Storage emulator."""
from __future__ import annotations
from core.platform.storage import cloud_storage_emulator
from typing import List, Optional, Union
CLIENT = cloud_storage_emulator.CloudStorageEmulator()

def isfile(unused_bucket_name: str, filepath: str) -> bool:
    if False:
        return 10
    'Checks if the file with the given filepath exists.\n\n    Args:\n        unused_bucket_name: str. Unused name of the GCS bucket.\n        filepath: str. The path to the relevant file.\n\n    Returns:\n        bool. Whether the file exists.\n    '
    return CLIENT.get_blob(filepath) is not None

def get(unused_bucket_name: str, filepath: str) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    'Gets a file data as bytes.\n\n    Args:\n        unused_bucket_name: str. Unused name of the GCS bucket.\n        filepath: str. The path to the relevant file.\n\n    Returns:\n        bytes. Returns data of the file as bytes.\n    '
    blob = CLIENT.get_blob(filepath)
    assert blob is not None
    return blob.download_as_bytes()

def commit(unused_bucket_name: str, filepath: str, raw_bytes: Union[bytes, str], mimetype: Optional[str]) -> None:
    if False:
        print('Hello World!')
    'Commits bytes to the relevant file.\n\n    Args:\n        unused_bucket_name: str. Unused name of the GCS bucket.\n        filepath: str. The path to the relevant file.\n        raw_bytes: bytes|str. The content to be stored in the file.\n        mimetype: Optional[str]. The content-type of the file.\n    '
    blob = cloud_storage_emulator.EmulatorBlob(filepath, raw_bytes, content_type=mimetype)
    CLIENT.upload_blob(filepath, blob)

def delete(unused_bucket_name: str, filepath: str) -> None:
    if False:
        while True:
            i = 10
    'Deletes a file and the metadata associated with it.\n\n    Args:\n        unused_bucket_name: str. Unused name of the GCS bucket.\n        filepath: str. The path to the relevant file.\n    '
    CLIENT.delete_blob(filepath)

def copy(unused_bucket_name: str, source_assets_path: str, dest_assets_path: str) -> None:
    if False:
        while True:
            i = 10
    "Copies images from source_path.\n\n    Args:\n        unused_bucket_name: str. Unused name of the GCS bucket.\n        source_assets_path: str. The path to the source entity's assets\n            folder.\n        dest_assets_path: str. The path to the relevant file within the entity's\n            assets folder.\n\n    Raises:\n        Exception. Source asset does not exist.\n    "
    src_blob = CLIENT.get_blob(source_assets_path)
    if src_blob is None:
        raise Exception('Source asset does not exist.')
    CLIENT.copy_blob(src_blob, dest_assets_path)

def listdir(unused_bucket_name: str, dir_name: str) -> List[cloud_storage_emulator.EmulatorBlob]:
    if False:
        return 10
    'Lists all files in a directory.\n\n    Args:\n        unused_bucket_name: str. Unused name of the GCS bucket.\n        dir_name: str. The directory whose files should be listed.\n\n    Returns:\n        list(EmulatorBlob). A lexicographically-sorted list of filenames.\n    '
    return CLIENT.list_blobs(dir_name)