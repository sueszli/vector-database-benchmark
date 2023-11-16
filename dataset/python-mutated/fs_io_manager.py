import os
import pickle
from typing import TYPE_CHECKING, Any, Optional
from pydantic import Field
import dagster._check as check
from dagster import DagsterInvariantViolationError, Field as DagsterField
from dagster._annotations import experimental
from dagster._config import StringSource
from dagster._config.pythonic_config import ConfigurableIOManagerFactory
from dagster._core.definitions.events import AssetKey, AssetMaterialization
from dagster._core.definitions.metadata import MetadataValue
from dagster._core.execution.context.init import InitResourceContext
from dagster._core.execution.context.input import InputContext
from dagster._core.execution.context.output import OutputContext
from dagster._core.storage.io_manager import IOManager, dagster_maintained_io_manager, io_manager
from dagster._core.storage.upath_io_manager import UPathIOManager
from dagster._utils import PICKLE_PROTOCOL, mkdir_p
if TYPE_CHECKING:
    from typing_extensions import Literal
    from upath import UPath

class FilesystemIOManager(ConfigurableIOManagerFactory['PickledObjectFilesystemIOManager']):
    """Built-in filesystem IO manager that stores and retrieves values using pickling.

    The base directory that the pickle files live inside is determined by:

    * The IO manager's "base_dir" configuration value, if specified. Otherwise...
    * A "storage/" directory underneath the value for "local_artifact_storage" in your dagster.yaml
      file, if specified. Otherwise...
    * A "storage/" directory underneath the directory that the DAGSTER_HOME environment variable
      points to, if that environment variable is specified. Otherwise...
    * A temporary directory.

    Assigns each op output to a unique filepath containing run ID, step key, and output name.
    Assigns each asset to a single filesystem path, at "<base_dir>/<asset_key>". If the asset key
    has multiple components, the final component is used as the name of the file, and the preceding
    components as parent directories under the base_dir.

    Subsequent materializations of an asset will overwrite previous materializations of that asset.
    So, with a base directory of "/my/base/path", an asset with key
    `AssetKey(["one", "two", "three"])` would be stored in a file called "three" in a directory
    with path "/my/base/path/one/two/".

    Example usage:


    1. Attach an IO manager to a set of assets using the reserved resource key ``"io_manager"``.

    .. code-block:: python

        from dagster import Definitions, asset, FilesystemIOManager

        @asset
        def asset1():
            # create df ...
            return df

        @asset
        def asset2(asset1):
            return asset1[:5]

        defs = Definitions(
            assets=[asset1, asset2],
            resources={
                "io_manager": FilesystemIOManager(base_dir="/my/base/path")
            },
        )


    2. Specify a job-level IO manager using the reserved resource key ``"io_manager"``,
    which will set the given IO manager on all ops in a job.

    .. code-block:: python

        from dagster import FilesystemIOManager, job, op

        @op
        def op_a():
            # create df ...
            return df

        @op
        def op_b(df):
            return df[:5]

        @job(
            resource_defs={
                "io_manager": FilesystemIOManager(base_dir="/my/base/path")
            }
        )
        def job():
            op_b(op_a())


    3. Specify IO manager on :py:class:`Out`, which allows you to set different IO managers on
    different step outputs.

    .. code-block:: python

        from dagster import FilesystemIOManager, job, op, Out

        @op(out=Out(io_manager_key="my_io_manager"))
        def op_a():
            # create df ...
            return df

        @op
        def op_b(df):
            return df[:5]

        @job(resource_defs={"my_io_manager": FilesystemIOManager()})
        def job():
            op_b(op_a())

    """
    base_dir: Optional[str] = Field(default=None, description='Base directory for storing files.')

    @classmethod
    def _is_dagster_maintained(cls) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def create_io_manager(self, context: InitResourceContext) -> 'PickledObjectFilesystemIOManager':
        if False:
            return 10
        base_dir = self.base_dir or check.not_none(context.instance).storage_directory()
        return PickledObjectFilesystemIOManager(base_dir=base_dir)

@dagster_maintained_io_manager
@io_manager(config_schema=FilesystemIOManager.to_config_schema(), description='Built-in filesystem IO manager that stores and retrieves values using pickling.')
def fs_io_manager(init_context: InitResourceContext) -> 'PickledObjectFilesystemIOManager':
    if False:
        for i in range(10):
            print('nop')
    'Built-in filesystem IO manager that stores and retrieves values using pickling.\n\n    The base directory that the pickle files live inside is determined by:\n\n    * The IO manager\'s "base_dir" configuration value, if specified. Otherwise...\n    * A "storage/" directory underneath the value for "local_artifact_storage" in your dagster.yaml\n      file, if specified. Otherwise...\n    * A "storage/" directory underneath the directory that the DAGSTER_HOME environment variable\n      points to, if that environment variable is specified. Otherwise...\n    * A temporary directory.\n\n    Assigns each op output to a unique filepath containing run ID, step key, and output name.\n    Assigns each asset to a single filesystem path, at "<base_dir>/<asset_key>". If the asset key\n    has multiple components, the final component is used as the name of the file, and the preceding\n    components as parent directories under the base_dir.\n\n    Subsequent materializations of an asset will overwrite previous materializations of that asset.\n    So, with a base directory of "/my/base/path", an asset with key\n    `AssetKey(["one", "two", "three"])` would be stored in a file called "three" in a directory\n    with path "/my/base/path/one/two/".\n\n    Example usage:\n\n\n    1. Attach an IO manager to a set of assets using the reserved resource key ``"io_manager"``.\n\n    .. code-block:: python\n\n        from dagster import Definitions, asset, fs_io_manager\n\n        @asset\n        def asset1():\n            # create df ...\n            return df\n\n        @asset\n        def asset2(asset1):\n            return asset1[:5]\n\n        defs = Definitions(\n            assets=[asset1, asset2],\n            resources={\n                "io_manager": fs_io_manager.configured({"base_dir": "/my/base/path"})\n            },\n        )\n\n\n    2. Specify a job-level IO manager using the reserved resource key ``"io_manager"``,\n    which will set the given IO manager on all ops in a job.\n\n    .. code-block:: python\n\n        from dagster import fs_io_manager, job, op\n\n        @op\n        def op_a():\n            # create df ...\n            return df\n\n        @op\n        def op_b(df):\n            return df[:5]\n\n        @job(\n            resource_defs={\n                "io_manager": fs_io_manager.configured({"base_dir": "/my/base/path"})\n            }\n        )\n        def job():\n            op_b(op_a())\n\n\n    3. Specify IO manager on :py:class:`Out`, which allows you to set different IO managers on\n    different step outputs.\n\n    .. code-block:: python\n\n        from dagster import fs_io_manager, job, op, Out\n\n        @op(out=Out(io_manager_key="my_io_manager"))\n        def op_a():\n            # create df ...\n            return df\n\n        @op\n        def op_b(df):\n            return df[:5]\n\n        @job(resource_defs={"my_io_manager": fs_io_manager})\n        def job():\n            op_b(op_a())\n\n    '
    return FilesystemIOManager.from_resource_context(init_context)

class PickledObjectFilesystemIOManager(UPathIOManager):
    """Built-in filesystem IO manager that stores and retrieves values using pickling.
    Is compatible with local and remote filesystems via `universal-pathlib` and `fsspec`.
    Learn more about how to use remote filesystems here: https://github.com/fsspec/universal_pathlib.

    Args:
        base_dir (Optional[str]): base directory where all the step outputs which use this object
            manager will be stored in.
        **kwargs: additional keyword arguments for `universal_pathlib.UPath`.
    """
    extension: str = ''

    def __init__(self, base_dir=None, **kwargs):
        if False:
            while True:
                i = 10
        from upath import UPath
        self.base_dir = check.opt_str_param(base_dir, 'base_dir')
        super().__init__(base_path=UPath(base_dir, **kwargs))

    def dump_to_path(self, context: OutputContext, obj: Any, path: 'UPath'):
        if False:
            print('Hello World!')
        try:
            with path.open('wb') as file:
                pickle.dump(obj, file, PICKLE_PROTOCOL)
        except (AttributeError, RecursionError, ImportError, pickle.PicklingError) as e:
            executor = context.step_context.job_def.executor_def
            if isinstance(e, RecursionError):
                obj_repr = f'{obj.__class__} exceeds recursion limit and'
            else:
                obj_repr = obj.__str__()
            raise DagsterInvariantViolationError(f'Object {obj_repr} is not picklable. You are currently using the fs_io_manager and the {executor.name}. You will need to use a different io manager to continue using this output. For example, you can use the mem_io_manager with the in_process_executor.\nFor more information on io managers, visit https://docs.dagster.io/concepts/io-management/io-managers \nFor more information on executors, vist https://docs.dagster.io/deployment/executors#overview') from e

    def load_from_path(self, context: InputContext, path: 'UPath') -> Any:
        if False:
            print('Hello World!')
        with path.open('rb') as file:
            return pickle.load(file)

class CustomPathPickledObjectFilesystemIOManager(IOManager):
    """Built-in filesystem IO managerthat stores and retrieves values using pickling and
    allow users to specify file path for outputs.

    Args:
        base_dir (Optional[str]): base directory where all the step outputs which use this object
            manager will be stored in.
    """

    def __init__(self, base_dir: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        self.base_dir = check.opt_str_param(base_dir, 'base_dir')
        self.write_mode: Literal['wb'] = 'wb'
        self.read_mode: Literal['rb'] = 'rb'

    def _get_path(self, path: str) -> str:
        if False:
            return 10
        return os.path.join(self.base_dir, path)

    def handle_output(self, context: OutputContext, obj: object):
        if False:
            while True:
                i = 10
        'Pickle the data and store the object to a custom file path.\n\n        This method emits an AssetMaterialization event so the assets will be tracked by the\n        Asset Catalog.\n        '
        check.inst_param(context, 'context', OutputContext)
        metadata = context.metadata
        path = check.str_param(metadata.get('path'), 'metadata.path')
        filepath = self._get_path(path)
        mkdir_p(os.path.dirname(filepath))
        context.log.debug(f'Writing file at: {filepath}')
        with open(filepath, self.write_mode) as write_obj:
            pickle.dump(obj, write_obj, PICKLE_PROTOCOL)
        return AssetMaterialization(asset_key=AssetKey([context.job_name, context.step_key, context.name]), metadata={'path': MetadataValue.path(os.path.abspath(filepath))})

    def load_input(self, context: InputContext) -> object:
        if False:
            for i in range(10):
                print('nop')
        'Unpickle the file from a given file path and Load it to a data object.'
        check.inst_param(context, 'context', InputContext)
        metadata = context.upstream_output.metadata
        path = check.str_param(metadata.get('path'), 'metadata.path')
        filepath = self._get_path(path)
        context.log.debug(f'Loading file from: {filepath}')
        with open(filepath, self.read_mode) as read_obj:
            return pickle.load(read_obj)

@dagster_maintained_io_manager
@io_manager(config_schema={'base_dir': DagsterField(StringSource, is_required=True)})
@experimental
def custom_path_fs_io_manager(init_context: InitResourceContext) -> CustomPathPickledObjectFilesystemIOManager:
    if False:
        return 10
    'Built-in IO manager that allows users to custom output file path per output definition.\n\n    It requires users to specify a base directory where all the step output will be stored in. It\n    serializes and deserializes output values (assets) using pickling and stores the pickled object\n    in the user-provided file paths.\n\n    Example usage:\n\n    .. code-block:: python\n\n        from dagster import custom_path_fs_io_manager, job, op\n\n        @op(out=Out(metadata={"path": "path/to/sample_output"}))\n        def sample_data(df):\n            return df[:5]\n\n        my_custom_path_fs_io_manager = custom_path_fs_io_manager.configured(\n            {"base_dir": "path/to/basedir"}\n        )\n\n        @job(resource_defs={"io_manager": my_custom_path_fs_io_manager})\n        def my_job():\n            sample_data()\n\n    '
    return CustomPathPickledObjectFilesystemIOManager(base_dir=init_context.resource_config.get('base_dir'))