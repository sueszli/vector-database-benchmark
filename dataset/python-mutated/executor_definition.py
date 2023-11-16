from enum import Enum as PyEnum
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Union, overload
from typing_extensions import Self, TypeAlias
import dagster._check as check
from dagster._annotations import public
from dagster._builtins import Int
from dagster._config import Field, Noneable, Selector, UserConfigSchema
from dagster._core.definitions.configurable import ConfiguredDefinitionConfigSchema, NamedConfigurableDefinition
from dagster._core.definitions.job_base import IJob
from dagster._core.definitions.reconstruct import ReconstructableJob
from dagster._core.errors import DagsterUnmetExecutorRequirementsError
from dagster._core.execution.retries import RetryMode, get_retries_config
from dagster._core.execution.tags import get_tag_concurrency_limits_config
from .definition_config_schema import IDefinitionConfigSchema, convert_user_facing_definition_config_schema
if TYPE_CHECKING:
    from dagster._core.executor.base import Executor
    from dagster._core.executor.in_process import InProcessExecutor
    from dagster._core.executor.init import InitExecutorContext
    from dagster._core.executor.multiprocess import MultiprocessExecutor
    from dagster._core.instance import DagsterInstance

class ExecutorRequirement(PyEnum):
    """An ExecutorDefinition can include a list of requirements that the system uses to
    check whether the executor will be able to work for a particular job execution.
    """
    RECONSTRUCTABLE_PIPELINE = 'RECONSTRUCTABLE_PIPELINE'
    RECONSTRUCTABLE_JOB = 'RECONSTRUCTABLE_PIPELINE'
    NON_EPHEMERAL_INSTANCE = 'NON_EPHEMERAL_INSTANCE'
    PERSISTENT_OUTPUTS = 'PERSISTENT_OUTPUTS'

def multiple_process_executor_requirements() -> Sequence[ExecutorRequirement]:
    if False:
        print('Hello World!')
    return [ExecutorRequirement.RECONSTRUCTABLE_JOB, ExecutorRequirement.NON_EPHEMERAL_INSTANCE, ExecutorRequirement.PERSISTENT_OUTPUTS]
ExecutorConfig = Mapping[str, object]
ExecutorCreationFunction: TypeAlias = Callable[['InitExecutorContext'], 'Executor']
ExecutorRequirementsFunction: TypeAlias = Callable[[ExecutorConfig], Sequence[ExecutorRequirement]]

class ExecutorDefinition(NamedConfigurableDefinition):
    """An executor is responsible for executing the steps of a job.

    Args:
        name (str): The name of the executor.
        config_schema (Optional[ConfigSchema]): The schema for the config. Configuration data
            available in `init_context.executor_config`. If not set, Dagster will accept any config
            provided.
        requirements (Optional[List[ExecutorRequirement]]): Any requirements that must
            be met in order for the executor to be usable for a particular job execution.
        executor_creation_fn(Optional[Callable]): Should accept an :py:class:`InitExecutorContext`
            and return an instance of :py:class:`Executor`
        required_resource_keys (Optional[Set[str]]): Keys for the resources required by the
            executor.
        description (Optional[str]): A description of the executor.
    """

    def __init__(self, name: str, config_schema: Optional[UserConfigSchema]=None, requirements: Union[ExecutorRequirementsFunction, Optional[Sequence[ExecutorRequirement]]]=None, executor_creation_fn: Optional[ExecutorCreationFunction]=None, description: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        self._name = check.str_param(name, 'name')
        self._requirements_fn: ExecutorRequirementsFunction
        if callable(requirements):
            self._requirements_fn = requirements
        else:
            requirements_lst = check.opt_list_param(requirements, 'requirements', of_type=ExecutorRequirement)
            self._requirements_fn = lambda _: requirements_lst
        self._config_schema = convert_user_facing_definition_config_schema(config_schema)
        self._executor_creation_fn = check.opt_callable_param(executor_creation_fn, 'executor_creation_fn')
        self._description = check.opt_str_param(description, 'description')

    @public
    @property
    def name(self) -> str:
        if False:
            while True:
                i = 10
        'Name of the executor.'
        return self._name

    @public
    @property
    def description(self) -> Optional[str]:
        if False:
            return 10
        'Description of executor, if provided.'
        return self._description

    @property
    def config_schema(self) -> IDefinitionConfigSchema:
        if False:
            return 10
        return self._config_schema

    def get_requirements(self, executor_config: Mapping[str, object]) -> Sequence[ExecutorRequirement]:
        if False:
            return 10
        return self._requirements_fn(executor_config)

    @public
    @property
    def executor_creation_fn(self) -> Optional[ExecutorCreationFunction]:
        if False:
            print('Hello World!')
        'Callable that takes an :py:class:`InitExecutorContext` and returns an instance of\n        :py:class:`Executor`.\n        '
        return self._executor_creation_fn

    def copy_for_configured(self, name, description, config_schema) -> 'ExecutorDefinition':
        if False:
            return 10
        return ExecutorDefinition(name=name, config_schema=config_schema, executor_creation_fn=self.executor_creation_fn, description=description or self.description, requirements=self._requirements_fn)

    @staticmethod
    def hardcoded_executor(executor: 'Executor'):
        if False:
            for i in range(10):
                print('nop')
        return ExecutorDefinition(name='__executor__', executor_creation_fn=lambda _init_context: executor)

    @public
    def configured(self, config_or_config_fn: Any, name: Optional[str]=None, config_schema: Optional[UserConfigSchema]=None, description: Optional[str]=None) -> Self:
        if False:
            while True:
                i = 10
        "Wraps this object in an object of the same type that provides configuration to the inner\n        object.\n\n        Using ``configured`` may result in config values being displayed in\n        the Dagster UI, so it is not recommended to use this API with sensitive values,\n        such as secrets.\n\n        Args:\n            config_or_config_fn (Union[Any, Callable[[Any], Any]]): Either (1) Run configuration\n                that fully satisfies this object's config schema or (2) A function that accepts run\n                configuration and returns run configuration that fully satisfies this object's\n                config schema.  In the latter case, config_schema must be specified.  When\n                passing a function, it's easiest to use :py:func:`configured`.\n            name (Optional[str]): Name of the new definition. If not provided, the emitted\n                definition will inherit the name of the `ExecutorDefinition` upon which this\n                function is called.\n            config_schema (Optional[ConfigSchema]): If config_or_config_fn is a function, the config\n                schema that its input must satisfy. If not set, Dagster will accept any config\n                provided.\n            description (Optional[str]): Description of the new definition. If not specified,\n                inherits the description of the definition being configured.\n\n        Returns (ConfigurableDefinition): A configured version of this object.\n        "
        name = check.opt_str_param(name, 'name')
        new_config_schema = ConfiguredDefinitionConfigSchema(self, convert_user_facing_definition_config_schema(config_schema), config_or_config_fn)
        return self.copy_for_configured(name or self.name, description, new_config_schema)

@overload
def executor(name: ExecutorCreationFunction) -> ExecutorDefinition:
    if False:
        return 10
    ...

@overload
def executor(name: Optional[str]=..., config_schema: Optional[UserConfigSchema]=..., requirements: Optional[Union[ExecutorRequirementsFunction, Sequence[ExecutorRequirement]]]=...) -> '_ExecutorDecoratorCallable':
    if False:
        for i in range(10):
            print('nop')
    ...

def executor(name: Union[ExecutorCreationFunction, Optional[str]]=None, config_schema: Optional[UserConfigSchema]=None, requirements: Optional[Union[ExecutorRequirementsFunction, Sequence[ExecutorRequirement]]]=None) -> Union[ExecutorDefinition, '_ExecutorDecoratorCallable']:
    if False:
        while True:
            i = 10
    'Define an executor.\n\n    The decorated function should accept an :py:class:`InitExecutorContext` and return an instance\n    of :py:class:`Executor`.\n\n    Args:\n        name (Optional[str]): The name of the executor.\n        config_schema (Optional[ConfigSchema]): The schema for the config. Configuration data available in\n            `init_context.executor_config`. If not set, Dagster will accept any config provided for.\n        requirements (Optional[List[ExecutorRequirement]]): Any requirements that must\n            be met in order for the executor to be usable for a particular job execution.\n    '
    if callable(name):
        check.invariant(config_schema is None)
        check.invariant(requirements is None)
        return _ExecutorDecoratorCallable()(name)
    return _ExecutorDecoratorCallable(name=name, config_schema=config_schema, requirements=requirements)

class _ExecutorDecoratorCallable:

    def __init__(self, name=None, config_schema=None, requirements=None):
        if False:
            print('Hello World!')
        self.name = check.opt_str_param(name, 'name')
        self.config_schema = config_schema
        self.requirements = requirements

    def __call__(self, fn: ExecutorCreationFunction) -> ExecutorDefinition:
        if False:
            while True:
                i = 10
        check.callable_param(fn, 'fn')
        if not self.name:
            self.name = fn.__name__
        executor_def = ExecutorDefinition(name=self.name, config_schema=self.config_schema, executor_creation_fn=fn, requirements=self.requirements)
        update_wrapper(executor_def, wrapped=fn)
        return executor_def

def _core_in_process_executor_creation(config: ExecutorConfig) -> 'InProcessExecutor':
    if False:
        while True:
            i = 10
    from dagster._core.executor.in_process import InProcessExecutor
    return InProcessExecutor(retries=RetryMode.from_config(check.dict_elem(config, 'retries')), marker_to_close=config.get('marker_to_close'))
IN_PROC_CONFIG = Field({'retries': get_retries_config(), 'marker_to_close': Field(str, is_required=False, description='[DEPRECATED]')}, description='Execute all steps in a single process.')

@executor(name='in_process', config_schema=IN_PROC_CONFIG)
def in_process_executor(init_context):
    if False:
        print('Hello World!')
    'The in-process executor executes all steps in a single process.\n\n    To select it, include the following top-level fragment in config:\n\n    .. code-block:: yaml\n\n        execution:\n          in_process:\n\n    Execution priority can be configured using the ``dagster/priority`` tag via op metadata,\n    where the higher the number the higher the priority. 0 is the default and both positive\n    and negative numbers can be used.\n    '
    return _core_in_process_executor_creation(init_context.executor_config)

@executor(name='execute_in_process_executor')
def execute_in_process_executor(_) -> 'InProcessExecutor':
    if False:
        while True:
            i = 10
    'Executor used by execute_in_process.\n\n    Use of this executor triggers special behavior in the config system that ignores all incoming\n    executor config. This is because someone might set executor config on a job, and when we foist\n    this executor onto the job for `execute_in_process`, that config becomes nonsensical.\n    '
    from dagster._core.executor.in_process import InProcessExecutor
    return InProcessExecutor(retries=RetryMode.ENABLED, marker_to_close=None)

def _core_multiprocess_executor_creation(config: ExecutorConfig) -> 'MultiprocessExecutor':
    if False:
        print('Hello World!')
    from dagster._core.executor.multiprocess import MultiprocessExecutor
    start_method = None
    start_cfg: Dict[str, object] = {}
    start_selector = check.opt_dict_elem(config, 'start_method')
    if start_selector:
        (start_method, start_cfg) = next(iter(start_selector.items()))
    return MultiprocessExecutor(max_concurrent=check.opt_int_elem(config, 'max_concurrent'), tag_concurrency_limits=check.opt_list_elem(config, 'tag_concurrency_limits'), retries=RetryMode.from_config(check.dict_elem(config, 'retries')), start_method=start_method, explicit_forkserver_preload=check.opt_list_elem(start_cfg, 'preload_modules', of_type=str))
MULTI_PROC_CONFIG = Field({'max_concurrent': Field(Noneable(Int), default_value=None, description='The number of processes that may run concurrently. By default, this is set to be the return value of `multiprocessing.cpu_count()`.'), 'tag_concurrency_limits': get_tag_concurrency_limits_config(), 'start_method': Field(Selector(fields={'spawn': Field({}, description='Configure the multiprocess executor to start subprocesses using `spawn`.'), 'forkserver': Field({'preload_modules': Field([str], is_required=False, description='Explicitly specify the modules to preload in the forkserver. Otherwise, there are two cases for default values if modules are not specified. If the Dagster job was loaded from a module, the same module will be preloaded. If not, the `dagster` module is preloaded.')}, description='Configure the multiprocess executor to start subprocesses using `forkserver`.')}), is_required=False, description='Select how subprocesses are created. By default, `spawn` is selected. See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods.'), 'retries': get_retries_config()}, description='Execute each step in an individual process.')

@executor(name='multiprocess', config_schema=MULTI_PROC_CONFIG, requirements=multiple_process_executor_requirements())
def multiprocess_executor(init_context):
    if False:
        for i in range(10):
            print('nop')
    'The multiprocess executor executes each step in an individual process.\n\n    Any job that does not specify custom executors will use the multiprocess_executor by default.\n    To configure the multiprocess executor, include a fragment such as the following in your run\n    config:\n\n    .. code-block:: yaml\n\n        execution:\n          config:\n            multiprocess:\n              max_concurrent: 4\n\n    The ``max_concurrent`` arg is optional and tells the execution engine how many processes may run\n    concurrently. By default, or if you set ``max_concurrent`` to be None or 0, this is the return value of\n    :py:func:`python:multiprocessing.cpu_count`.\n\n    Execution priority can be configured using the ``dagster/priority`` tag via op metadata,\n    where the higher the number the higher the priority. 0 is the default and both positive\n    and negative numbers can be used.\n    '
    return _core_multiprocess_executor_creation(init_context.executor_config)

def check_cross_process_constraints(init_context: 'InitExecutorContext') -> None:
    if False:
        for i in range(10):
            print('nop')
    from dagster._core.executor.init import InitExecutorContext
    check.inst_param(init_context, 'init_context', InitExecutorContext)
    requirements_lst = init_context.executor_def.get_requirements(init_context.executor_config)
    if ExecutorRequirement.RECONSTRUCTABLE_JOB in requirements_lst:
        _check_intra_process_job(init_context.job)
    if ExecutorRequirement.NON_EPHEMERAL_INSTANCE in requirements_lst:
        _check_non_ephemeral_instance(init_context.instance)

def _check_intra_process_job(job: IJob) -> None:
    if False:
        return 10
    if not isinstance(job, ReconstructableJob):
        raise DagsterUnmetExecutorRequirementsError(f'You have attempted to use an executor that uses multiple processes with the job "{job.get_definition().name}" that is not reconstructable. Job must be loaded in a way that allows dagster to reconstruct them in a new process. This means: \n  * using the file, module, or workspace.yaml arguments of dagster-webserver/dagster-graphql/dagster\n  * loading the job through the reconstructable() function\n')

def _check_non_ephemeral_instance(instance: 'DagsterInstance') -> None:
    if False:
        for i in range(10):
            print('nop')
    if instance.is_ephemeral:
        raise DagsterUnmetExecutorRequirementsError('You have attempted to use an executor that uses multiple processes with an ephemeral DagsterInstance. A non-ephemeral instance is needed to coordinate execution between multiple processes. You can configure your default instance via $DAGSTER_HOME or ensure a valid one is passed when invoking the python APIs. You can learn more about setting up a persistent DagsterInstance from the DagsterInstance docs here: https://docs.dagster.io/deployment/dagster-instance#default-local-behavior')

def _get_default_executor_requirements(executor_config: ExecutorConfig) -> Sequence[ExecutorRequirement]:
    if False:
        return 10
    return multiple_process_executor_requirements() if 'multiprocess' in executor_config else []

@executor(name='multi_or_in_process_executor', config_schema=Field(Selector({'multiprocess': MULTI_PROC_CONFIG, 'in_process': IN_PROC_CONFIG}), default_value={'multiprocess': {}}), requirements=_get_default_executor_requirements)
def multi_or_in_process_executor(init_context: 'InitExecutorContext') -> 'Executor':
    if False:
        print('Hello World!')
    'The default executor for a job.\n\n    This is the executor available by default on a :py:class:`JobDefinition`\n    that does not provide custom executors. This executor has a multiprocessing-enabled mode, and a\n    single-process mode. By default, multiprocessing mode is enabled. Switching between multiprocess\n    mode and in-process mode can be achieved via config.\n\n    .. code-block:: yaml\n\n        execution:\n          config:\n            multiprocess:\n\n\n        execution:\n          config:\n            in_process:\n\n    When using the multiprocess mode, ``max_concurrent`` and ``retries`` can also be configured.\n\n    .. code-block:: yaml\n\n        execution:\n          config:\n            multiprocess:\n              max_concurrent: 4\n              retries:\n                enabled:\n\n    The ``max_concurrent`` arg is optional and tells the execution engine how many processes may run\n    concurrently. By default, or if you set ``max_concurrent`` to be 0, this is the return value of\n    :py:func:`python:multiprocessing.cpu_count`.\n\n    When using the in_process mode, then only retries can be configured.\n\n    Execution priority can be configured using the ``dagster/priority`` tag via op metadata,\n    where the higher the number the higher the priority. 0 is the default and both positive\n    and negative numbers can be used.\n    '
    if 'multiprocess' in init_context.executor_config:
        return _core_multiprocess_executor_creation(check.dict_elem(init_context.executor_config, 'multiprocess'))
    else:
        return _core_in_process_executor_creation(check.dict_elem(init_context.executor_config, 'in_process'))