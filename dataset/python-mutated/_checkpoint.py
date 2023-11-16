import contextlib
import glob
import json
import logging
import os
import platform
import shutil
import tempfile
import traceback
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union
import pyarrow.fs
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.storage import _download_from_fs_path, _exists_at_fs_path
from ray.util.annotations import PublicAPI
logger = logging.getLogger(__name__)
_METADATA_FILE_NAME = '.metadata.json'
_CHECKPOINT_TEMP_DIR_PREFIX = 'checkpoint_tmp_'

class _CheckpointMetaClass(type):

    def __getattr__(self, item):
        if False:
            i = 10
            return i + 15
        try:
            return super().__getattribute__(item)
        except AttributeError as exc:
            if item in {'from_dict', 'to_dict', 'from_bytes', 'to_bytes', 'get_internal_representation'}:
                raise _get_migration_error(item) from exc
            elif item in {'from_uri', 'to_uri', 'uri'}:
                raise _get_uri_error(item) from exc
            elif item in {'get_preprocessor', 'set_preprocessor'}:
                raise _get_preprocessor_error(item) from exc
            raise exc

@PublicAPI(stability='beta')
class Checkpoint(metaclass=_CheckpointMetaClass):
    """A reference to data persisted as a directory in local or remote storage.

    Access the checkpoint contents locally using ``checkpoint.to_directory()``
    or ``checkpoint.as_directory``.

    Example creating a checkpoint using ``Checkpoint.from_directory``:

        >>> from ray.train import Checkpoint
        >>> checkpoint = Checkpoint.from_directory("/tmp/example_checkpoint_dir")
        >>> checkpoint.filesystem  # doctest: +ELLIPSIS
        <pyarrow._fs.LocalFileSystem object...
        >>> checkpoint.path
        '/tmp/example_checkpoint_dir'

    Example creating a checkpoint from a remote URI:

        >>> checkpoint = Checkpoint("s3://bucket/path/to/checkpoint")
        >>> checkpoint.filesystem  # doctest: +ELLIPSIS
        <pyarrow._s3fs.S3FileSystem object...
        >>> checkpoint.path
        'bucket/path/to/checkpoint'

    Example creating a checkpoint with a custom filesystem:

        >>> checkpoint = Checkpoint(
        ...     path="bucket/path/to/checkpoint",
        ...     filesystem=pyarrow.fs.S3FileSystem(),
        ... )
        >>> checkpoint.filesystem  # doctest: +ELLIPSIS
        <pyarrow._s3fs.S3FileSystem object...
        >>> checkpoint.path
        'bucket/path/to/checkpoint'

    Attributes:
        path: A path on the filesystem containing the checkpoint contents.
        filesystem: PyArrow FileSystem that can be used to access data at the `path`.
    """

    def __init__(self, path: Union[str, os.PathLike], filesystem: Optional['pyarrow.fs.FileSystem']=None):
        if False:
            print('Hello World!')
        'Construct a Checkpoint.\n\n        Args:\n            path: A local path or remote URI containing the checkpoint data.\n                If a filesystem is provided, then this path must NOT be a URI.\n                It should be a path on the filesystem with the prefix already stripped.\n            filesystem: PyArrow FileSystem to use to access data at the path.\n                If not specified, this is inferred from the URI scheme.\n        '
        self.path = str(path)
        self.filesystem = filesystem
        if path and (not filesystem):
            (self.filesystem, self.path) = pyarrow.fs.FileSystem.from_uri(path)
        self._uuid = uuid.uuid4()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'Checkpoint(filesystem={self.filesystem.type_name}, path={self.path})'

    def get_metadata(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Return the metadata dict stored with the checkpoint.\n\n        If no metadata is stored, an empty dict is returned.\n        '
        metadata_path = os.path.join(self.path, _METADATA_FILE_NAME)
        if not _exists_at_fs_path(self.filesystem, metadata_path):
            return {}
        with self.filesystem.open_input_file(metadata_path) as f:
            return json.loads(f.readall().decode('utf-8'))

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        'Set the metadata stored with this checkpoint.\n\n        This will overwrite any existing metadata stored with this checkpoint.\n        '
        metadata_path = os.path.join(self.path, _METADATA_FILE_NAME)
        with self.filesystem.open_output_stream(metadata_path) as f:
            f.write(json.dumps(metadata).encode('utf-8'))

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        if False:
            return 10
        'Update the metadata stored with this checkpoint.\n\n        This will update any existing metadata stored with this checkpoint.\n        '
        existing_metadata = self.get_metadata()
        existing_metadata.update(metadata)
        self.set_metadata(existing_metadata)

    @classmethod
    def from_directory(cls, path: Union[str, os.PathLike]) -> 'Checkpoint':
        if False:
            i = 10
            return i + 15
        'Create checkpoint object from a local directory.\n\n        Args:\n            path: Local directory containing checkpoint data.\n\n        Returns:\n            A ray.train.Checkpoint object.\n        '
        return cls(path, filesystem=pyarrow.fs.LocalFileSystem())

    def to_directory(self, path: Optional[Union[str, os.PathLike]]=None) -> str:
        if False:
            return 10
        'Write checkpoint data to a local directory.\n\n        *If multiple processes on the same node call this method simultaneously,*\n        only a single process will perform the download, while the others\n        wait for the download to finish. Once the download finishes, all processes\n        receive the same local directory to read from.\n\n        Args:\n            path: Target directory to download data to. If not specified,\n                this method will use a temporary directory.\n\n        Returns:\n            str: Directory containing checkpoint data.\n        '
        user_provided_path = path is not None
        local_path = path if user_provided_path else self._get_temporary_checkpoint_dir()
        local_path = os.path.normpath(os.path.expanduser(str(local_path)))
        os.makedirs(local_path, exist_ok=True)
        try:
            with TempFileLock(local_path, timeout=0):
                _download_from_fs_path(fs=self.filesystem, fs_path=self.path, local_path=local_path)
        except TimeoutError:
            with TempFileLock(local_path, timeout=-1):
                pass
            if not os.path.exists(local_path):
                raise RuntimeError(f'Checkpoint directory {local_path} does not exist, even though it should have been created by another process. Please raise an issue on GitHub: https://github.com/ray-project/ray/issues')
        return local_path

    @contextlib.contextmanager
    def as_directory(self) -> Iterator[str]:
        if False:
            while True:
                i = 10
        'Returns checkpoint contents in a local directory as a context.\n\n        This function makes checkpoint data available as a directory while avoiding\n        unnecessary copies and left-over temporary data.\n\n        *If the checkpoint points to a local directory*, this method just returns the\n        local directory path without making a copy, and nothing will be cleaned up\n        after exiting the context.\n\n        *If the checkpoint points to a remote directory*, this method will download the\n        checkpoint to a local temporary directory and return the path\n        to the temporary directory.\n\n        *If multiple processes on the same node call this method simultaneously,*\n        only a single process will perform the download, while the others\n        wait for the download to finish. Once the download finishes, all processes\n        receive the same local (temporary) directory to read from.\n\n        Once all processes have finished working with the checkpoint,\n        the temporary directory is cleaned up.\n\n        Users should treat the returned checkpoint directory as read-only and avoid\n        changing any data within it, as it may be deleted when exiting the context.\n\n        Example:\n\n        .. testcode::\n            :hide:\n\n            from pathlib import Path\n            import tempfile\n\n            from ray.train import Checkpoint\n\n            temp_dir = tempfile.mkdtemp()\n            (Path(temp_dir) / "example.txt").write_text("example checkpoint data")\n            checkpoint = Checkpoint.from_directory(temp_dir)\n\n        .. testcode::\n\n            with checkpoint.as_directory() as checkpoint_dir:\n                # Do some read-only processing of files within checkpoint_dir\n                pass\n\n            # At this point, if a temporary directory was created, it will have\n            # been deleted.\n\n        '
        if isinstance(self.filesystem, pyarrow.fs.LocalFileSystem):
            yield self.path
        else:
            del_lock_path = _get_del_lock_path(self._get_temporary_checkpoint_dir())
            open(del_lock_path, 'a').close()
            temp_dir = self.to_directory()
            try:
                yield temp_dir
            finally:
                try:
                    os.remove(del_lock_path)
                except Exception:
                    logger.warning(f'Could not remove {del_lock_path} deletion file lock. Traceback:\n{traceback.format_exc()}')
                remaining_locks = _list_existing_del_locks(temp_dir)
                if not remaining_locks:
                    try:
                        with TempFileLock(temp_dir, timeout=0):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                    except TimeoutError:
                        pass

    def _get_temporary_checkpoint_dir(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return the name for the temporary checkpoint dir that this checkpoint\n        will get downloaded to, if accessing via `to_directory` or `as_directory`.\n        '
        tmp_dir_path = tempfile.gettempdir()
        checkpoint_dir_name = _CHECKPOINT_TEMP_DIR_PREFIX + self._uuid.hex
        if platform.system() == 'Windows':
            del_lock_name = _get_del_lock_path('')
            checkpoint_dir_name = _CHECKPOINT_TEMP_DIR_PREFIX + self._uuid.hex[-259 + len(_CHECKPOINT_TEMP_DIR_PREFIX) + len(tmp_dir_path) + len(del_lock_name):]
            if not checkpoint_dir_name.startswith(_CHECKPOINT_TEMP_DIR_PREFIX):
                raise RuntimeError("Couldn't create checkpoint directory due to length constraints. Try specifying a shorter checkpoint path.")
        return os.path.join(tmp_dir_path, checkpoint_dir_name)

    def __fspath__(self):
        if False:
            print('Hello World!')
        raise TypeError('You cannot use `Checkpoint` objects directly as paths. Use `Checkpoint.to_directory()` or `Checkpoint.as_directory()` instead.')

def _get_del_lock_path(path: str, suffix: str=None) -> str:
    if False:
        i = 10
        return i + 15
    'Get the path to the deletion lock file for a file/directory at `path`.\n\n    Example:\n\n        >>> _get_del_lock_path("/tmp/checkpoint_tmp")  # doctest: +ELLIPSIS\n        \'/tmp/checkpoint_tmp.del_lock_...\n        >>> _get_del_lock_path("/tmp/checkpoint_tmp/")  # doctest: +ELLIPSIS\n        \'/tmp/checkpoint_tmp.del_lock_...\n        >>> _get_del_lock_path("/tmp/checkpoint_tmp.txt")  # doctest: +ELLIPSIS\n        \'/tmp/checkpoint_tmp.txt.del_lock_...\n\n    '
    suffix = suffix if suffix is not None else str(os.getpid())
    return f"{path.rstrip('/')}.del_lock_{suffix}"

def _list_existing_del_locks(path: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'List all the deletion lock files for a file/directory at `path`.\n\n    For example, if 2 checkpoints are being read via `as_directory`,\n    then this should return a list of 2 deletion lock files.\n    '
    return list(glob.glob(f"{_get_del_lock_path(path, suffix='*')}"))

def _get_migration_error(name: str):
    if False:
        return 10
    return AttributeError(f"The new `ray.train.Checkpoint` class does not support `{name}()`. Instead, only directories are supported.\n\nExample to store a dictionary in a checkpoint:\n\nimport os, tempfile\nimport ray.cloudpickle as pickle\nfrom ray import train\nfrom ray.train import Checkpoint\n\nwith tempfile.TemporaryDirectory() as checkpoint_dir:\n  with open(os.path.join(checkpoint_dir, 'data.pkl'), 'wb') as fp:\n    pickle.dump({{'data': 'value'}}, fp)\n\n  checkpoint = Checkpoint.from_directory(checkpoint_dir)\n  train.report(..., checkpoint=checkpoint)\n\nExample to load a dictionary from a checkpoint:\n\nif train.get_checkpoint():\n  with train.get_checkpoint().as_directory() as checkpoint_dir:\n    with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:\n      data = pickle.load(fp)")

def _get_uri_error(name: str):
    if False:
        i = 10
        return i + 15
    return AttributeError(f'The new `ray.train.Checkpoint` class does not support `{name}()`. To create a checkpoint from remote storage, create a `Checkpoint` using its constructor instead of `from_directory`.\nExample: `Checkpoint(path="s3://a/b/c")`.\nThen, access the contents of the checkpoint with `checkpoint.as_directory()` / `checkpoint.to_directory()`.\nTo upload data to remote storage, use e.g. `pyarrow.fs.FileSystem` or your client of choice.')

def _get_preprocessor_error(name: str):
    if False:
        i = 10
        return i + 15
    return AttributeError(f'The new `ray.train.Checkpoint` class does not support `{name}()`. To include preprocessor information in checkpoints, pass it as metadata in the <Framework>Trainer constructor.\nExample: `TorchTrainer(..., metadata={{...}})`.\nAfter training, access it in the checkpoint via `checkpoint.get_metadata()`. See here: https://docs.ray.io/en/master/train/user-guides/data-loading-preprocessing.html#preprocessing-structured-data')