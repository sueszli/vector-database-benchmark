import asyncio
import inspect
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from dagster import InputContext, MetadataValue, MultiPartitionKey, OutputContext, _check as check
from dagster._core.storage.memoizable_io_manager import MemoizableIOManager
if TYPE_CHECKING:
    from upath import UPath

class UPathIOManager(MemoizableIOManager):
    """Abstract IOManager base class compatible with local and cloud storage via `universal-pathlib` and `fsspec`.

    Features:
     - handles partitioned assets
     - handles loading a single upstream partition
     - handles loading multiple upstream partitions (with respect to :py:class:`PartitionMapping`)
     - supports loading multiple partitions concurrently with async `load_from_path` method
     - the `get_metadata` method can be customized to add additional metadata to the output
     - the `allow_missing_partitions` metadata value can be set to `True` to skip missing partitions
       (the default behavior is to raise an error)

    """
    extension: Optional[str] = None

    def __init__(self, base_path: Optional['UPath']=None):
        if False:
            return 10
        from upath import UPath
        assert not self.extension or '.' in self.extension
        self._base_path = base_path or UPath('.')

    @abstractmethod
    def dump_to_path(self, context: OutputContext, obj: Any, path: 'UPath'):
        if False:
            i = 10
            return i + 15
        'Child classes should override this method to write the object to the filesystem.'

    @abstractmethod
    def load_from_path(self, context: InputContext, path: 'UPath') -> Any:
        if False:
            i = 10
            return i + 15
        'Child classes should override this method to load the object from the filesystem.'

    @property
    def fs(self) -> AbstractFileSystem:
        if False:
            while True:
                i = 10
        'Utility function to get the IOManager filesystem.\n\n        Returns:\n            AbstractFileSystem: fsspec filesystem.\n\n        '
        from upath import UPath
        if isinstance(self._base_path, UPath):
            return self._base_path.fs
        elif isinstance(self._base_path, Path):
            return LocalFileSystem()
        else:
            raise ValueError(f'Unsupported base_path type: {type(self._base_path)}')

    @property
    def storage_options(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Utility function to get the fsspec storage_options which are often consumed by various I/O functions.\n\n        Returns:\n            Dict[str, Any]: fsspec storage_options.\n        '
        from upath import UPath
        if isinstance(self._base_path, UPath):
            return self._base_path._kwargs.copy()
        elif isinstance(self._base_path, Path):
            return {}
        else:
            raise ValueError(f'Unsupported base_path type: {type(self._base_path)}')

    def get_metadata(self, context: OutputContext, obj: Any) -> Dict[str, MetadataValue]:
        if False:
            while True:
                i = 10
        'Child classes should override this method to add custom metadata to the outputs.'
        return {}

    def unlink(self, path: 'UPath') -> None:
        if False:
            i = 10
            return i + 15
        'Remove the file or object at the provided path.'
        path.unlink()

    def path_exists(self, path: 'UPath') -> bool:
        if False:
            print('Hello World!')
        'Check if a file or object exists at the provided path.'
        return path.exists()

    def make_directory(self, path: 'UPath'):
        if False:
            while True:
                i = 10
        "Create a directory at the provided path.\n\n        Override as a no-op if the target backend doesn't use directories.\n        "
        path.mkdir(parents=True, exist_ok=True)

    def has_output(self, context: OutputContext) -> bool:
        if False:
            i = 10
            return i + 15
        return self.path_exists(self._get_path(context))

    def _with_extension(self, path: 'UPath') -> 'UPath':
        if False:
            for i in range(10):
                print('nop')
        return path.with_suffix(path.suffix + self.extension) if self.extension else path

    def _get_path_without_extension(self, context: Union[InputContext, OutputContext]) -> 'UPath':
        if False:
            while True:
                i = 10
        if context.has_asset_key:
            context_path = self.get_asset_relative_path(context)
        else:
            context_path = self.get_op_output_relative_path(context)
        return self._base_path.joinpath(context_path)

    def get_asset_relative_path(self, context: Union[InputContext, OutputContext]) -> 'UPath':
        if False:
            print('Hello World!')
        from upath import UPath
        return UPath(*context.asset_key.path)

    def get_op_output_relative_path(self, context: Union[InputContext, OutputContext]) -> 'UPath':
        if False:
            return 10
        from upath import UPath
        return UPath(*context.get_identifier())

    def get_loading_input_log_message(self, path: 'UPath') -> str:
        if False:
            while True:
                i = 10
        return f'Loading file from: {path} using {self.__class__.__name__}...'

    def get_writing_output_log_message(self, path: 'UPath') -> str:
        if False:
            print('Hello World!')
        return f'Writing file at: {path} using {self.__class__.__name__}...'

    def get_loading_input_partition_log_message(self, path: 'UPath', partition_key: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'Loading partition {partition_key} from {path} using {self.__class__.__name__}...'

    def get_missing_partition_log_message(self, partition_key: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f"Couldn't load partition {partition_key} and skipped it because the input metadata includes allow_missing_partitions=True"

    def _get_path(self, context: Union[InputContext, OutputContext]) -> 'UPath':
        if False:
            print('Hello World!')
        'Returns the I/O path for a given context.\n        Should not be used with partitions (use `_get_paths_for_partitions` instead).\n        '
        path = self._get_path_without_extension(context)
        return self._with_extension(path)

    def get_path_for_partition(self, context: Union[InputContext, OutputContext], path: 'UPath', partition: str) -> 'UPath':
        if False:
            print('Hello World!')
        'Override this method if you want to use a different partitioning scheme\n        (for example, if the saving function handles partitioning instead).\n        The extension will be added later.\n\n        Args:\n            context (Union[InputContext, OutputContext]): The context for the I/O operation.\n            path (UPath): The path to the file or object.\n            partition (str): Formatted partition/multipartition key\n\n        Returns:\n            UPath: The path to the file with the partition key appended.\n        '
        return path / partition

    def _get_paths_for_partitions(self, context: Union[InputContext, OutputContext]) -> Dict[str, 'UPath']:
        if False:
            i = 10
            return i + 15
        'Returns a dict of partition_keys into I/O paths for a given context.'
        if not context.has_asset_partitions:
            raise TypeError(f'Detected {context.dagster_type.typing_type} input type but the asset is not partitioned')

        def _formatted_multipartitioned_path(partition_key: MultiPartitionKey) -> str:
            if False:
                print('Hello World!')
            ordered_dimension_keys = [key[1] for key in sorted(partition_key.keys_by_dimension.items(), key=lambda x: x[0])]
            return '/'.join(ordered_dimension_keys)
        formatted_partition_keys = {partition_key: _formatted_multipartitioned_path(partition_key) if isinstance(partition_key, MultiPartitionKey) else partition_key for partition_key in context.asset_partition_keys}
        asset_path = self._get_path_without_extension(context)
        return {partition_key: self._with_extension(self.get_path_for_partition(context, asset_path, partition)) for (partition_key, partition) in formatted_partition_keys.items()}

    def _get_multipartition_backcompat_paths(self, context: Union[InputContext, OutputContext]) -> Mapping[str, 'UPath']:
        if False:
            for i in range(10):
                print('nop')
        if not context.has_asset_partitions:
            raise TypeError(f'Detected {context.dagster_type.typing_type} input type but the asset is not partitioned')
        partition_keys = context.asset_partition_keys
        asset_path = self._get_path_without_extension(context)
        return {partition_key: self._with_extension(asset_path / partition_key) for partition_key in partition_keys if isinstance(partition_key, MultiPartitionKey)}

    def _load_single_input(self, path: 'UPath', context: InputContext, backcompat_path: Optional['UPath']=None) -> Any:
        if False:
            return 10
        context.log.debug(self.get_loading_input_log_message(path))
        try:
            obj = self.load_from_path(context=context, path=path)
            if asyncio.iscoroutine(obj):
                obj = asyncio.run(obj)
        except FileNotFoundError as e:
            if backcompat_path is not None:
                try:
                    obj = self.load_from_path(context=context, path=backcompat_path)
                    if asyncio.iscoroutine(obj):
                        obj = asyncio.run(obj)
                    context.log.debug(f'File not found at {path}. Loaded instead from backcompat path: {backcompat_path}')
                except FileNotFoundError:
                    raise e
            else:
                raise e
        context.add_input_metadata({'path': MetadataValue.path(str(path))})
        return obj

    def _load_partition_from_path(self, context: InputContext, partition_key: str, path: 'UPath', backcompat_path: Optional['UPath']=None) -> Any:
        if False:
            print('Hello World!')
        '1. Try to load the partition from the normal path.\n        2. If it was not found, try to load it from the backcompat path.\n        3. If allow_missing_partitions metadata is True, skip the partition if it was not found in any of the paths.\n        Otherwise, raise an error.\n\n        Args:\n            context (InputContext): IOManager Input context\n            partition_key (str): the partition key corresponding to the partition being loaded\n            path (UPath): The path to the partition.\n            backcompat_path (Optional[UPath]): The path to the partition in the backcompat location.\n\n        Returns:\n            Any: The object loaded from the partition.\n        '
        allow_missing_partitions = context.metadata.get('allow_missing_partitions', False) if context.metadata is not None else False
        try:
            context.log.debug(self.get_loading_input_partition_log_message(path, partition_key))
            obj = self.load_from_path(context=context, path=path)
            return obj
        except FileNotFoundError as e:
            if backcompat_path is not None:
                try:
                    obj = self.load_from_path(context=context, path=path)
                    context.log.debug(f'File not found at {path}. Loaded instead from backcompat path: {backcompat_path}')
                    return obj
                except FileNotFoundError as e:
                    if allow_missing_partitions:
                        context.log.warning(self.get_missing_partition_log_message(partition_key))
                        return None
                    else:
                        raise e
            if allow_missing_partitions:
                context.log.warning(self.get_missing_partition_log_message(partition_key))
                return None
            else:
                raise e

    def _load_multiple_inputs(self, context: InputContext) -> Dict[str, Any]:
        if False:
            return 10
        paths = self._get_paths_for_partitions(context)
        backcompat_paths = self._get_multipartition_backcompat_paths(context)
        context.log.debug(f'Loading {len(paths)} partitions...')
        objs = {}
        if not inspect.iscoroutinefunction(self.load_from_path):
            for partition_key in context.asset_partition_keys:
                obj = self._load_partition_from_path(context, partition_key, paths[partition_key], backcompat_paths.get(partition_key))
                if obj is not None:
                    objs[partition_key] = obj
            return objs
        else:

            async def collect():
                loop = asyncio.get_running_loop()
                tasks = []
                for partition_key in context.asset_partition_keys:
                    tasks.append(loop.create_task(self._load_partition_from_path(context, partition_key, paths[partition_key], backcompat_paths.get(partition_key))))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                allow_missing_partitions = context.metadata.get('allow_missing_partitions', False) if context.metadata is not None else False
                results_without_errors = []
                found_errors = False
                for (partition_key, result) in zip(context.asset_partition_keys, results):
                    if isinstance(result, FileNotFoundError):
                        if allow_missing_partitions:
                            context.log.warning(self.get_missing_partition_log_message(partition_key))
                        else:
                            context.log.error(str(result))
                            found_errors = True
                    elif isinstance(result, Exception):
                        context.log.error(str(result))
                        found_errors = True
                    else:
                        results_without_errors.append(result)
                if found_errors:
                    raise RuntimeError(f'{len(paths) - len(results_without_errors)} partitions could not be loaded')
                return results_without_errors
            awaited_objects = asyncio.get_event_loop().run_until_complete(collect())
            return {partition_key: awaited_object for (partition_key, awaited_object) in zip(context.asset_partition_keys, awaited_objects) if awaited_object is not None}

    def load_input(self, context: InputContext) -> Union[Any, Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        if not context.has_asset_key or not context.has_asset_partitions:
            path = self._get_path(context)
            return self._load_single_input(path, context)
        else:
            asset_partition_keys = context.asset_partition_keys
            if len(asset_partition_keys) == 0:
                return None
            elif len(asset_partition_keys) == 1:
                paths = self._get_paths_for_partitions(context)
                check.invariant(len(paths) == 1, f'Expected 1 path, but got {len(paths)}')
                path = next(iter(paths.values()))
                backcompat_paths = self._get_multipartition_backcompat_paths(context)
                backcompat_path = None if not backcompat_paths else next(iter(backcompat_paths.values()))
                return self._load_single_input(path, context, backcompat_path)
            else:
                type_annotation = context.dagster_type.typing_type
                if type_annotation != Any and (not is_dict_type(type_annotation)):
                    check.failed(f"Loading an input that corresponds to multiple partitions, but the type annotation on the op input is not a dict, Dict, Mapping, or Any: is '{type_annotation}'.")
                return self._load_multiple_inputs(context)

    def handle_output(self, context: OutputContext, obj: Any):
        if False:
            print('Hello World!')
        if context.dagster_type.typing_type == type(None):
            check.invariant(obj is None, f"Output had Nothing type or 'None' annotation, but handle_output received value that was not None and was of type {type(obj)}.")
            return None
        if context.has_asset_partitions:
            paths = self._get_paths_for_partitions(context)
            check.invariant(len(paths) == 1, f"The current IO manager {type(self)} does not support persisting an output associated with multiple partitions. This error is likely occurring because a backfill was launched using the 'single run' option. Instead, launch the backfill with the 'multiple runs' option.")
            path = next(iter(paths.values()))
        else:
            path = self._get_path(context)
        self.make_directory(path.parent)
        context.log.debug(self.get_writing_output_log_message(path))
        self.dump_to_path(context=context, obj=obj, path=path)
        metadata = {'path': MetadataValue.path(str(path))}
        custom_metadata = self.get_metadata(context=context, obj=obj)
        metadata.update(custom_metadata)
        context.add_output_metadata(metadata)

def is_dict_type(type_obj) -> bool:
    if False:
        return 10
    if type_obj == dict:
        return True
    if hasattr(type_obj, '__origin__') and type_obj.__origin__ in (dict, Dict, Mapping):
        return True
    return False