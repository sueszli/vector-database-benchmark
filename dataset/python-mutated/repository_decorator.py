from functools import update_wrapper
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, TypeVar, Union, overload
from typing_extensions import TypeAlias
import dagster._check as check
from dagster._core.decorator_utils import get_function_params
from dagster._core.definitions.metadata import RawMetadataValue, normalize_metadata
from dagster._core.definitions.resource_definition import ResourceDefinition
from dagster._core.errors import DagsterInvalidDefinitionError
from ..asset_checks import AssetChecksDefinition
from ..executor_definition import ExecutorDefinition
from ..graph_definition import GraphDefinition
from ..job_definition import JobDefinition
from ..logger_definition import LoggerDefinition
from ..partitioned_schedule import UnresolvedPartitionedAssetScheduleDefinition
from ..repository_definition import VALID_REPOSITORY_DATA_DICT_KEYS, CachingRepositoryData, PendingRepositoryDefinition, PendingRepositoryListDefinition, RepositoryData, RepositoryDefinition, RepositoryListDefinition
from ..schedule_definition import ScheduleDefinition
from ..sensor_definition import SensorDefinition
from ..unresolved_asset_job_definition import UnresolvedAssetJobDefinition
T = TypeVar('T')
RepositoryDictSpec: TypeAlias = Dict[str, Dict[str, RepositoryListDefinition]]

def _flatten(items: Iterable[Union[T, List[T]]]) -> Iterator[T]:
    if False:
        return 10
    for x in items:
        if isinstance(x, List):
            yield from x
        else:
            yield x

class _Repository:

    def __init__(self, name: Optional[str]=None, description: Optional[str]=None, metadata: Optional[Dict[str, RawMetadataValue]]=None, default_executor_def: Optional[ExecutorDefinition]=None, default_logger_defs: Optional[Mapping[str, LoggerDefinition]]=None, top_level_resources: Optional[Mapping[str, ResourceDefinition]]=None, resource_key_mapping: Optional[Mapping[int, str]]=None):
        if False:
            for i in range(10):
                print('nop')
        self.name = check.opt_str_param(name, 'name')
        self.description = check.opt_str_param(description, 'description')
        self.metadata = normalize_metadata(check.opt_mapping_param(metadata, 'metadata', key_type=str))
        self.default_executor_def = check.opt_inst_param(default_executor_def, 'default_executor_def', ExecutorDefinition)
        self.default_logger_defs = check.opt_mapping_param(default_logger_defs, 'default_logger_defs', key_type=str, value_type=LoggerDefinition)
        self.top_level_resources = check.opt_mapping_param(top_level_resources, 'top_level_resources', key_type=str, value_type=ResourceDefinition)
        self.resource_key_mapping = check.opt_mapping_param(resource_key_mapping, 'resource_key_mapping', key_type=int, value_type=str)

    @overload
    def __call__(self, fn: Union[Callable[[], Sequence[RepositoryListDefinition]], Callable[[], RepositoryDictSpec]]) -> RepositoryDefinition:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __call__(self, fn: Callable[[], Sequence[PendingRepositoryListDefinition]]) -> PendingRepositoryDefinition:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __call__(self, fn: Union[Callable[[], Sequence[PendingRepositoryListDefinition]], Callable[[], RepositoryDictSpec]]) -> Union[RepositoryDefinition, PendingRepositoryDefinition]:
        if False:
            return 10
        from dagster._core.definitions import AssetsDefinition, SourceAsset
        from dagster._core.definitions.cacheable_assets import CacheableAssetsDefinition
        check.callable_param(fn, 'fn')
        if not self.name:
            self.name = fn.__name__
        repository_definitions = fn()
        repository_data: Optional[Union[CachingRepositoryData, RepositoryData]]
        if isinstance(repository_definitions, list):
            bad_defns = []
            repository_defns = []
            defer_repository_data = False
            for (i, definition) in enumerate(_flatten(repository_definitions)):
                if isinstance(definition, CacheableAssetsDefinition):
                    defer_repository_data = True
                elif not isinstance(definition, (JobDefinition, ScheduleDefinition, UnresolvedPartitionedAssetScheduleDefinition, SensorDefinition, GraphDefinition, AssetsDefinition, SourceAsset, UnresolvedAssetJobDefinition, AssetChecksDefinition)):
                    bad_defns.append((i, type(definition)))
                else:
                    repository_defns.append(definition)
            if bad_defns:
                bad_definitions_str = ', '.join([f'value of type {type_} at index {i}' for (i, type_) in bad_defns])
                raise DagsterInvalidDefinitionError(f'Bad return value from repository construction function: all elements of list must be of type JobDefinition, GraphDefinition, ScheduleDefinition, SensorDefinition, AssetsDefinition, SourceAsset, or AssetChecksDefinition.Got {bad_definitions_str}.')
            repository_data = None if defer_repository_data else CachingRepositoryData.from_list(repository_defns, default_executor_def=self.default_executor_def, default_logger_defs=self.default_logger_defs, top_level_resources=self.top_level_resources, resource_key_mapping=self.resource_key_mapping)
        elif isinstance(repository_definitions, dict):
            if not set(repository_definitions.keys()).issubset(VALID_REPOSITORY_DATA_DICT_KEYS):
                raise DagsterInvalidDefinitionError("Bad return value from repository construction function: dict must not contain keys other than {{'schedules', 'sensors', 'jobs'}}: found {bad_keys}".format(bad_keys=', '.join([f"'{key}'" for key in repository_definitions.keys() if key not in VALID_REPOSITORY_DATA_DICT_KEYS])))
            repository_data = CachingRepositoryData.from_dict(repository_definitions)
        elif isinstance(repository_definitions, RepositoryData):
            repository_data = repository_definitions
        else:
            raise DagsterInvalidDefinitionError('Bad return value of type {type_} from repository construction function: must return list, dict, or RepositoryData. See the @repository decorator docstring for details and examples'.format(type_=type(repository_definitions)))
        if isinstance(repository_definitions, list) and repository_data is None:
            return PendingRepositoryDefinition(self.name, repository_definitions=list(_flatten(repository_definitions)), description=self.description, metadata=self.metadata, default_executor_def=self.default_executor_def, default_logger_defs=self.default_logger_defs, _top_level_resources=self.top_level_resources)
        else:
            repository_def = RepositoryDefinition(name=self.name, description=self.description, metadata=self.metadata, repository_data=repository_data)
            update_wrapper(repository_def, fn)
            return repository_def

@overload
def repository(definitions_fn: Union[Callable[[], Sequence[RepositoryListDefinition]], Callable[[], RepositoryDictSpec]]) -> RepositoryDefinition:
    if False:
        i = 10
        return i + 15
    ...

@overload
def repository(definitions_fn: Callable[..., Sequence[PendingRepositoryListDefinition]]) -> PendingRepositoryDefinition:
    if False:
        print('Hello World!')
    ...

@overload
def repository(*, name: Optional[str]=..., description: Optional[str]=..., metadata: Optional[Dict[str, RawMetadataValue]]=..., default_executor_def: Optional[ExecutorDefinition]=..., default_logger_defs: Optional[Mapping[str, LoggerDefinition]]=..., _top_level_resources: Optional[Mapping[str, ResourceDefinition]]=..., _resource_key_mapping: Optional[Mapping[int, str]]=...) -> _Repository:
    if False:
        print('Hello World!')
    ...

def repository(definitions_fn: Optional[Union[Callable[[], Sequence[PendingRepositoryListDefinition]], Callable[[], RepositoryDictSpec]]]=None, *, name: Optional[str]=None, description: Optional[str]=None, metadata: Optional[Dict[str, RawMetadataValue]]=None, default_executor_def: Optional[ExecutorDefinition]=None, default_logger_defs: Optional[Mapping[str, LoggerDefinition]]=None, _top_level_resources: Optional[Mapping[str, ResourceDefinition]]=None, _resource_key_mapping: Optional[Mapping[int, str]]=None) -> Union[RepositoryDefinition, PendingRepositoryDefinition, _Repository]:
    if False:
        return 10
    'Create a repository from the decorated function.\n\n    The decorated function should take no arguments and its return value should one of:\n\n    1. ``List[Union[JobDefinition, ScheduleDefinition, SensorDefinition]]``.\n    Use this form when you have no need to lazy load jobs or other definitions. This is the\n    typical use case.\n\n    2. A dict of the form:\n\n    .. code-block:: python\n\n        {\n            \'jobs\': Dict[str, Callable[[], JobDefinition]],\n            \'schedules\': Dict[str, Callable[[], ScheduleDefinition]]\n            \'sensors\': Dict[str, Callable[[], SensorDefinition]]\n        }\n\n    This form is intended to allow definitions to be created lazily when accessed by name,\n    which can be helpful for performance when there are many definitions in a repository, or\n    when constructing the definitions is costly.\n\n    3. A :py:class:`RepositoryData`. Return this object if you need fine-grained\n    control over the construction and indexing of definitions within the repository, e.g., to\n    create definitions dynamically from .yaml files in a directory.\n\n    Args:\n        name (Optional[str]): The name of the repository. Defaults to the name of the decorated\n            function.\n        description (Optional[str]): A string description of the repository.\n        metadata (Optional[Dict[str, RawMetadataValue]]): Arbitrary metadata for the repository.\n        top_level_resources (Optional[Mapping[str, ResourceDefinition]]): A dict of top-level\n            resource keys to defintions, for resources which should be displayed in the UI.\n\n    Example:\n        .. code-block:: python\n\n            ######################################################################\n            # A simple repository using the first form of the decorated function\n            ######################################################################\n\n            @op(config_schema={n: Field(Int)})\n            def return_n(context):\n                return context.op_config[\'n\']\n\n            @job\n            def simple_job():\n                return_n()\n\n            @job\n            def some_job():\n                ...\n\n            @sensor(job=some_job)\n            def some_sensor():\n                if foo():\n                    yield RunRequest(\n                        run_key= ...,\n                        run_config={\n                            \'ops\': {\'return_n\': {\'config\': {\'n\': bar()}}}\n                        }\n                    )\n\n            @job\n            def my_job():\n                ...\n\n            my_schedule = ScheduleDefinition(cron_schedule="0 0 * * *", job=my_job)\n\n            @repository\n            def simple_repository():\n                return [simple_job, some_sensor, my_schedule]\n\n            ######################################################################\n            # A simple repository using the first form of the decorated function\n            # and custom metadata that will be displayed in the UI\n            ######################################################################\n\n            ...\n\n            @repository(\n                name=\'my_repo\',\n                metadata={\n                    \'team\': \'Team A\',\n                    \'repository_version\': \'1.2.3\',\n                    \'environment\': \'production\',\n             })\n            def simple_repository():\n                return [simple_job, some_sensor, my_schedule]\n\n            ######################################################################\n            # A lazy-loaded repository\n            ######################################################################\n\n            def make_expensive_job():\n                @job\n                def expensive_job():\n                    for i in range(10000):\n                        return_n.alias(f\'return_n_{i}\')()\n\n                return expensive_job\n\n            def make_expensive_schedule():\n                @job\n                def other_expensive_job():\n                    for i in range(11000):\n                        return_n.alias(f\'my_return_n_{i}\')()\n\n                return ScheduleDefinition(cron_schedule="0 0 * * *", job=other_expensive_job)\n\n            @repository\n            def lazy_loaded_repository():\n                return {\n                    \'jobs\': {\'expensive_job\': make_expensive_job},\n                    \'schedules\': {\'expensive_schedule\': make_expensive_schedule}\n                }\n\n\n            ######################################################################\n            # A complex repository that lazily constructs jobs from a directory\n            # of files in a bespoke YAML format\n            ######################################################################\n\n            class ComplexRepositoryData(RepositoryData):\n                def __init__(self, yaml_directory):\n                    self._yaml_directory = yaml_directory\n\n                def get_all_jobs(self):\n                    return [\n                        self._construct_job_def_from_yaml_file(\n                          self._yaml_file_for_job_name(file_name)\n                        )\n                        for file_name in os.listdir(self._yaml_directory)\n                    ]\n\n                ...\n\n            @repository\n            def complex_repository():\n                return ComplexRepositoryData(\'some_directory\')\n    '
    if definitions_fn is not None:
        check.invariant(description is None)
        check.invariant(len(get_function_params(definitions_fn)) == 0)
        return _Repository()(definitions_fn)
    return _Repository(name=name, description=description, metadata=metadata, default_executor_def=default_executor_def, default_logger_defs=default_logger_defs, top_level_resources=_top_level_resources, resource_key_mapping=_resource_key_mapping)