"""Utilities related to data saving/loading."""
import io
import logging
from pathlib import Path
from typing import IO, Any, Dict, Union
import fsspec
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem
from lightning_utilities.core.imports import module_available
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
log = logging.getLogger(__name__)

def _load(path_or_url: Union[IO, _PATH], map_location: _MAP_LOCATION_TYPE=None) -> Any:
    if False:
        print('Hello World!')
    'Loads a checkpoint.\n\n    Args:\n        path_or_url: Path or URL of the checkpoint.\n        map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.\n\n    '
    if not isinstance(path_or_url, (str, Path)):
        return torch.load(path_or_url, map_location=map_location)
    if str(path_or_url).startswith('http'):
        return torch.hub.load_state_dict_from_url(str(path_or_url), map_location=map_location)
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, 'rb') as f:
        return torch.load(f, map_location=map_location)

def get_filesystem(path: _PATH, **kwargs: Any) -> AbstractFileSystem:
    if False:
        for i in range(10):
            print('nop')
    (fs, _) = url_to_fs(str(path), **kwargs)
    return fs

def _atomic_save(checkpoint: Dict[str, Any], filepath: Union[str, Path]) -> None:
    if False:
        print('Hello World!')
    'Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.\n\n    Args:\n        checkpoint: The object to save.\n            Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``\n            accepts.\n        filepath: The path to which the checkpoint will be saved.\n            This points to the file that the checkpoint will be stored in.\n\n    '
    bytesbuffer = io.BytesIO()
    log.debug(f'Saving checkpoint: {filepath}')
    torch.save(checkpoint, bytesbuffer)
    with fsspec.open(filepath, 'wb') as f:
        f.write(bytesbuffer.getvalue())

def _is_object_storage(fs: AbstractFileSystem) -> bool:
    if False:
        i = 10
        return i + 15
    if module_available('adlfs'):
        from adlfs import AzureBlobFileSystem
        if isinstance(fs, AzureBlobFileSystem):
            return True
    if module_available('gcsfs'):
        from gcsfs import GCSFileSystem
        if isinstance(fs, GCSFileSystem):
            return True
    if module_available('s3fs'):
        from s3fs import S3FileSystem
        if isinstance(fs, S3FileSystem):
            return True
    return False

def _is_dir(fs: AbstractFileSystem, path: Union[str, Path], strict: bool=False) -> bool:
    if False:
        return 10
    'Check if a path is directory-like.\n\n    This function determines if a given path is considered directory-like, taking into account the behavior\n    specific to object storage platforms. For other filesystems, it behaves similarly to the standard `fs.isdir`\n    method.\n\n    Args:\n        fs: The filesystem to check the path against.\n        path: The path or URL to be checked.\n        strict: A flag specific to Object Storage platforms. If set to ``False``, any non-existing path is considered\n            as a valid directory-like path. In such cases, the directory (and any non-existing parent directories)\n            will be created on the fly. Defaults to False.\n\n    '
    if _is_object_storage(fs):
        if strict:
            return fs.isdir(path)
        return not fs.isfile(path)
    return fs.isdir(path)