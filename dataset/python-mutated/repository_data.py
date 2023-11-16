from abc import ABC, abstractmethod
from types import FunctionType
from typing import TYPE_CHECKING, AbstractSet, Any, Callable, Dict, Mapping, Optional, Sequence, TypeVar, Union
import dagster._check as check
from dagster._annotations import public
from dagster._core.definitions.events import AssetKey
from dagster._core.definitions.executor_definition import ExecutorDefinition
from dagster._core.definitions.graph_definition import SubselectedGraphDefinition
from dagster._core.definitions.job_definition import JobDefinition
from dagster._core.definitions.logger_definition import LoggerDefinition
from dagster._core.definitions.resource_definition import ResourceDefinition
from dagster._core.definitions.schedule_definition import ScheduleDefinition
from dagster._core.definitions.sensor_definition import SensorDefinition
from dagster._core.definitions.source_asset import SourceAsset
from dagster._core.errors import DagsterInvalidDefinitionError, DagsterInvariantViolationError
from .caching_index import CacheingDefinitionIndex
from .valid_definitions import RepositoryListDefinition
if TYPE_CHECKING:
    from dagster._core.definitions import AssetsDefinition
T = TypeVar('T')
Resolvable = Callable[[], T]

class RepositoryData(ABC):
    """Users should usually rely on the :py:func:`@repository <repository>` decorator to create new
    repositories, which will in turn call the static constructors on this class. However, users may
    subclass :py:class:`RepositoryData` for fine-grained control over access to and lazy creation
    of repository members.
    """

    @abstractmethod
    def get_resource_key_mapping(self) -> Mapping[int, str]:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def get_top_level_resources(self) -> Mapping[str, ResourceDefinition]:
        if False:
            print('Hello World!')
        'Return all top-level resources in the repository as a list,\n        such as those provided to the Definitions constructor.\n\n        Returns:\n            List[ResourceDefinition]: All top-level resources in the repository.\n        '

    @abstractmethod
    def get_env_vars_by_top_level_resource(self) -> Mapping[str, AbstractSet[str]]:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    @public
    def get_all_jobs(self) -> Sequence[JobDefinition]:
        if False:
            for i in range(10):
                print('nop')
        'Return all jobs in the repository as a list.\n\n        Returns:\n            List[JobDefinition]: All jobs in the repository.\n        '

    @public
    def get_job_names(self) -> Sequence[str]:
        if False:
            i = 10
            return i + 15
        'Get the names of all jobs in the repository.\n\n        Returns:\n            List[str]\n        '
        return [job_def.name for job_def in self.get_all_jobs()]

    @public
    def has_job(self, job_name: str) -> bool:
        if False:
            while True:
                i = 10
        'Check if a job with a given name is present in the repository.\n\n        Args:\n            job_name (str): The name of the job.\n\n        Returns:\n            bool\n        '
        return job_name in self.get_job_names()

    @public
    def get_job(self, job_name: str) -> JobDefinition:
        if False:
            i = 10
            return i + 15
        'Get a job by name.\n\n        Args:\n            job_name (str): Name of the job to retrieve.\n\n        Returns:\n            JobDefinition: The job definition corresponding to the given name.\n        '
        match = next((job for job in self.get_all_jobs() if job.name == job_name))
        if match is None:
            raise DagsterInvariantViolationError(f'Could not find job {job_name} in repository')
        return match

    @public
    def get_schedule_names(self) -> Sequence[str]:
        if False:
            return 10
        'Get the names of all schedules in the repository.\n\n        Returns:\n            List[str]\n        '
        return [schedule.name for schedule in self.get_all_schedules()]

    @public
    def get_all_schedules(self) -> Sequence[ScheduleDefinition]:
        if False:
            print('Hello World!')
        'Return all schedules in the repository as a list.\n\n        Returns:\n            List[ScheduleDefinition]: All jobs in the repository.\n        '
        return []

    @public
    def get_schedule(self, schedule_name: str) -> ScheduleDefinition:
        if False:
            return 10
        'Get a schedule by name.\n\n        Args:\n            schedule_name (str): name of the schedule to retrieve.\n\n        Returns:\n            ScheduleDefinition: The schedule definition corresponding to the given name.\n        '
        schedules_with_name = [schedule for schedule in self.get_all_schedules() if schedule.name == schedule_name]
        if not schedules_with_name:
            raise DagsterInvariantViolationError(f'Could not find schedule {schedule_name} in repository')
        return schedules_with_name[0]

    @public
    def has_schedule(self, schedule_name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if a schedule with a given name is present in the repository.'
        return schedule_name in self.get_schedule_names()

    @public
    def get_all_sensors(self) -> Sequence[SensorDefinition]:
        if False:
            return 10
        'Sequence[SensorDefinition]: Return all sensors in the repository as a list.'
        return []

    @public
    def get_sensor_names(self) -> Sequence[str]:
        if False:
            for i in range(10):
                print('nop')
        'Sequence[str]: Get the names of all sensors in the repository.'
        return [sensor.name for sensor in self.get_all_sensors()]

    @public
    def get_sensor(self, sensor_name: str) -> SensorDefinition:
        if False:
            while True:
                i = 10
        'Get a sensor by name.\n\n        Args:\n            sensor_name (str): name of the sensor to retrieve.\n\n        Returns:\n            SensorDefinition: The sensor definition corresponding to the given name.\n        '
        sensors_with_name = [sensor for sensor in self.get_all_sensors() if sensor.name == sensor_name]
        if not sensors_with_name:
            raise DagsterInvariantViolationError(f'Could not find sensor {sensor_name} in repository')
        return sensors_with_name[0]

    @public
    def has_sensor(self, sensor_name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if a sensor with a given name is present in the repository.'
        return sensor_name in self.get_sensor_names()

    @public
    def get_source_assets_by_key(self) -> Mapping[AssetKey, SourceAsset]:
        if False:
            return 10
        'Mapping[AssetKey, SourceAsset]: Get the source assets for the repository.'
        return {}

    @public
    def get_assets_defs_by_key(self) -> Mapping[AssetKey, 'AssetsDefinition']:
        if False:
            return 10
        'Mapping[AssetKey, AssetsDefinition]: Get the asset definitions for the repository.'
        return {}

    def load_all_definitions(self):
        if False:
            print('Hello World!')
        self.get_all_jobs()
        self.get_all_schedules()
        self.get_all_sensors()
        self.get_source_assets_by_key()

class CachingRepositoryData(RepositoryData):
    """Default implementation of RepositoryData used by the :py:func:`@repository <repository>` decorator."""
    _all_jobs: Optional[Sequence[JobDefinition]]
    _all_pipelines: Optional[Sequence[JobDefinition]]

    def __init__(self, jobs: Mapping[str, Union[JobDefinition, Resolvable[JobDefinition]]], schedules: Mapping[str, Union[ScheduleDefinition, Resolvable[ScheduleDefinition]]], sensors: Mapping[str, Union[SensorDefinition, Resolvable[SensorDefinition]]], source_assets_by_key: Mapping[AssetKey, SourceAsset], assets_defs_by_key: Mapping[AssetKey, 'AssetsDefinition'], top_level_resources: Mapping[str, ResourceDefinition], utilized_env_vars: Mapping[str, AbstractSet[str]], resource_key_mapping: Mapping[int, str]):
        if False:
            while True:
                i = 10
        'Constructs a new CachingRepositoryData object.\n\n        You may pass pipeline, job, and schedule definitions directly, or you may pass callables\n        with no arguments that will be invoked to lazily construct definitions when accessed by\n        name. This can be helpful for performance when there are many definitions in a repository,\n        or when constructing the definitions is costly.\n\n        Note that when lazily constructing a definition, the name of the definition must match its\n        key in its dictionary index, or a :py:class:`DagsterInvariantViolationError` will be thrown\n        at retrieval time.\n\n        Args:\n            jobs (Mapping[str, Union[JobDefinition, Callable[[], JobDefinition]]]):\n                The job definitions belonging to the repository.\n            schedules (Mapping[str, Union[ScheduleDefinition, Callable[[], ScheduleDefinition]]]):\n                The schedules belonging to the repository.\n            sensors (Mapping[str, Union[SensorDefinition, Callable[[], SensorDefinition]]]):\n                The sensors belonging to a repository.\n            source_assets_by_key (Mapping[AssetKey, SourceAsset]): The source assets belonging to a repository.\n            assets_defs_by_key (Mapping[AssetKey, AssetsDefinition]): The assets definitions\n                belonging to a repository.\n            top_level_resources (Mapping[str, ResourceDefinition]): A dict of top-level\n                resource keys to defintions, for resources which should be displayed in the UI.\n        '
        from dagster._core.definitions import AssetsDefinition
        check.mapping_param(jobs, 'jobs', key_type=str, value_type=(JobDefinition, FunctionType))
        check.mapping_param(schedules, 'schedules', key_type=str, value_type=(ScheduleDefinition, FunctionType))
        check.mapping_param(sensors, 'sensors', key_type=str, value_type=(SensorDefinition, FunctionType))
        check.mapping_param(source_assets_by_key, 'source_assets_by_key', key_type=AssetKey, value_type=SourceAsset)
        check.mapping_param(assets_defs_by_key, 'assets_defs_by_key', key_type=AssetKey, value_type=AssetsDefinition)
        check.mapping_param(top_level_resources, 'top_level_resources', key_type=str, value_type=ResourceDefinition)
        check.mapping_param(utilized_env_vars, 'utilized_resources', key_type=str)
        check.mapping_param(resource_key_mapping, 'resource_key_mapping', key_type=int, value_type=str)
        self._jobs = CacheingDefinitionIndex(JobDefinition, 'JobDefinition', 'job', jobs, self._validate_job)
        self._schedules = CacheingDefinitionIndex(ScheduleDefinition, 'ScheduleDefinition', 'schedule', schedules, self._validate_schedule)
        self._schedules.get_all_definitions()
        self._source_assets_by_key = source_assets_by_key
        self._assets_defs_by_key = assets_defs_by_key
        self._top_level_resources = top_level_resources
        self._utilized_env_vars = utilized_env_vars
        self._resource_key_mapping = resource_key_mapping
        self._sensors = CacheingDefinitionIndex(SensorDefinition, 'SensorDefinition', 'sensor', sensors, self._validate_sensor)
        self._sensors.get_all_definitions()
        self._all_jobs = None

    @staticmethod
    def from_dict(repository_definitions: Dict[str, Dict[str, Any]]) -> 'CachingRepositoryData':
        if False:
            print('Hello World!')
        "Static constructor.\n\n        Args:\n            repository_definition (Dict[str, Dict[str, ...]]): A dict of the form:\n\n                {\n                    'jobs': Dict[str, Callable[[], JobDefinition]],\n                    'schedules': Dict[str, Callable[[], ScheduleDefinition]]\n                }\n\n            This form is intended to allow definitions to be created lazily when accessed by name,\n            which can be helpful for performance when there are many definitions in a repository, or\n            when constructing the definitions is costly.\n        "
        from .repository_data_builder import build_caching_repository_data_from_dict
        return build_caching_repository_data_from_dict(repository_definitions)

    @classmethod
    def from_list(cls, repository_definitions: Sequence[RepositoryListDefinition], default_executor_def: Optional[ExecutorDefinition]=None, default_logger_defs: Optional[Mapping[str, LoggerDefinition]]=None, top_level_resources: Optional[Mapping[str, ResourceDefinition]]=None, resource_key_mapping: Optional[Mapping[int, str]]=None) -> 'CachingRepositoryData':
        if False:
            i = 10
            return i + 15
        'Static constructor.\n\n        Args:\n            repository_definitions (List[Union[JobDefinition, ScheduleDefinition, SensorDefinition, GraphDefinition]]):\n                Use this constructor when you have no need to lazy load jobs or other definitions.\n            top_level_resources (Optional[Mapping[str, ResourceDefinition]]): A dict of top-level\n                resource keys to defintions, for resources which should be displayed in the UI.\n        '
        from .repository_data_builder import build_caching_repository_data_from_list
        return build_caching_repository_data_from_list(repository_definitions=repository_definitions, default_executor_def=default_executor_def, default_logger_defs=default_logger_defs, top_level_resources=top_level_resources, resource_key_mapping=resource_key_mapping)

    def get_env_vars_by_top_level_resource(self) -> Mapping[str, AbstractSet[str]]:
        if False:
            while True:
                i = 10
        return self._utilized_env_vars

    def get_resource_key_mapping(self) -> Mapping[int, str]:
        if False:
            return 10
        return self._resource_key_mapping

    def get_job_names(self) -> Sequence[str]:
        if False:
            while True:
                i = 10
        'Get the names of all jobs in the repository.\n\n        Returns:\n            List[str]\n        '
        return self._jobs.get_definition_names()

    def has_job(self, job_name: str) -> bool:
        if False:
            while True:
                i = 10
        'Check if a job with a given name is present in the repository.\n\n        Args:\n            job_name (str): The name of the job.\n\n        Returns:\n            bool\n        '
        check.str_param(job_name, 'job_name')
        return self._jobs.has_definition(job_name)

    def get_top_level_resources(self) -> Mapping[str, ResourceDefinition]:
        if False:
            i = 10
            return i + 15
        return self._top_level_resources

    def get_all_jobs(self) -> Sequence[JobDefinition]:
        if False:
            return 10
        'Return all jobs in the repository as a list.\n\n        Note that this will construct any job that has not yet been constructed.\n\n        Returns:\n            List[JobDefinition]: All jobs in the repository.\n        '
        if self._all_jobs is not None:
            return self._all_jobs
        self._all_jobs = self._jobs.get_all_definitions()
        self._check_node_defs(self._all_jobs)
        return self._all_jobs

    def get_job(self, job_name: str) -> JobDefinition:
        if False:
            return 10
        'Get a job by name.\n\n        If this job has not yet been constructed, only this job is constructed, and will\n        be cached for future calls.\n\n        Args:\n            job_name (str): Name of the job to retrieve.\n\n        Returns:\n            JobDefinition: The job definition corresponding to the given name.\n        '
        check.str_param(job_name, 'job_name')
        return self._jobs.get_definition(job_name)

    def get_schedule_names(self) -> Sequence[str]:
        if False:
            while True:
                i = 10
        'Get the names of all schedules in the repository.\n\n        Returns:\n            List[str]\n        '
        return self._schedules.get_definition_names()

    def get_all_schedules(self) -> Sequence[ScheduleDefinition]:
        if False:
            return 10
        'Return all schedules in the repository as a list.\n\n        Note that this will construct any schedule that has not yet been constructed.\n\n        Returns:\n            List[ScheduleDefinition]: All schedules in the repository.\n        '
        return self._schedules.get_all_definitions()

    def get_schedule(self, schedule_name: str) -> ScheduleDefinition:
        if False:
            i = 10
            return i + 15
        'Get a schedule by name.\n\n        if this schedule has not yet been constructed, only this schedule is constructed, and will\n        be cached for future calls.\n\n        Args:\n            schedule_name (str): name of the schedule to retrieve.\n\n        Returns:\n            ScheduleDefinition: The schedule definition corresponding to the given name.\n        '
        check.str_param(schedule_name, 'schedule_name')
        return self._schedules.get_definition(schedule_name)

    def has_schedule(self, schedule_name: str) -> bool:
        if False:
            while True:
                i = 10
        check.str_param(schedule_name, 'schedule_name')
        return self._schedules.has_definition(schedule_name)

    def get_all_sensors(self) -> Sequence[SensorDefinition]:
        if False:
            while True:
                i = 10
        return self._sensors.get_all_definitions()

    def get_sensor_names(self) -> Sequence[str]:
        if False:
            i = 10
            return i + 15
        return self._sensors.get_definition_names()

    def get_sensor(self, sensor_name: str) -> SensorDefinition:
        if False:
            while True:
                i = 10
        return self._sensors.get_definition(sensor_name)

    def has_sensor(self, sensor_name: str) -> bool:
        if False:
            i = 10
            return i + 15
        return self._sensors.has_definition(sensor_name)

    def get_source_assets_by_key(self) -> Mapping[AssetKey, SourceAsset]:
        if False:
            return 10
        return self._source_assets_by_key

    def get_assets_defs_by_key(self) -> Mapping[AssetKey, 'AssetsDefinition']:
        if False:
            i = 10
            return i + 15
        return self._assets_defs_by_key

    def _check_node_defs(self, job_defs: Sequence[JobDefinition]) -> None:
        if False:
            print('Hello World!')
        node_defs = {}
        node_to_job = {}
        for job_def in job_defs:
            for node_def in [*job_def.all_node_defs, job_def.graph]:
                if isinstance(node_def, SubselectedGraphDefinition):
                    break
                if node_def.name not in node_defs:
                    node_defs[node_def.name] = node_def
                    node_to_job[node_def.name] = job_def.name
                if node_defs[node_def.name] is not node_def:
                    (first_name, second_name) = sorted([node_to_job[node_def.name], job_def.name])
                    raise DagsterInvalidDefinitionError(f"Conflicting definitions found in repository with name '{node_def.name}'. Op/Graph definition names must be unique within a repository. {node_def.__class__.__name__} is defined in job '{first_name}' and in job '{second_name}'.")

    def _validate_job(self, job: JobDefinition) -> JobDefinition:
        if False:
            while True:
                i = 10
        return job

    def _validate_schedule(self, schedule: ScheduleDefinition) -> ScheduleDefinition:
        if False:
            print('Hello World!')
        job_names = self.get_job_names()
        if schedule.job_name not in job_names:
            raise DagsterInvalidDefinitionError(f'ScheduleDefinition "{schedule.name}" targets job "{schedule.job_name}" which was not found in this repository.')
        return schedule

    def _validate_sensor(self, sensor: SensorDefinition) -> SensorDefinition:
        if False:
            return 10
        job_names = self.get_job_names()
        if len(sensor.targets) == 0:
            return sensor
        for target in sensor.targets:
            if target.job_name not in job_names:
                raise DagsterInvalidDefinitionError(f'SensorDefinition "{sensor.name}" targets job "{sensor.job_name}" which was not found in this repository.')
        return sensor