from contextlib import ExitStack
from typing import AbstractSet, Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Set, Union, cast
import dagster._check as check
from dagster._core.definitions.assets import AssetsDefinition
from dagster._core.definitions.composition import PendingNodeInvocation
from dagster._core.definitions.decorators.op_decorator import DecoratedOpFunction
from dagster._core.definitions.dependency import Node, NodeHandle
from dagster._core.definitions.events import AssetMaterialization, AssetObservation, ExpectationResult, UserEvent
from dagster._core.definitions.hook_definition import HookDefinition
from dagster._core.definitions.job_definition import JobDefinition
from dagster._core.definitions.multi_dimensional_partitions import MultiPartitionsDefinition
from dagster._core.definitions.op_definition import OpDefinition
from dagster._core.definitions.partition_key_range import PartitionKeyRange
from dagster._core.definitions.resource_definition import IContainsGenerator, ResourceDefinition, Resources, ScopedResourcesBuilder
from dagster._core.definitions.resource_requirement import ensure_requirements_satisfied
from dagster._core.definitions.step_launcher import StepLauncher
from dagster._core.definitions.time_window_partitions import TimeWindow, TimeWindowPartitionsDefinition, has_one_dimension_time_window_partitioning
from dagster._core.errors import DagsterInvalidInvocationError, DagsterInvalidPropertyError, DagsterInvariantViolationError
from dagster._core.execution.build_resources import build_resources, wrap_resources_for_execution
from dagster._core.instance import DagsterInstance
from dagster._core.log_manager import DagsterLogManager
from dagster._core.storage.dagster_run import DagsterRun
from dagster._core.types.dagster_type import DagsterType
from dagster._utils.forked_pdb import ForkedPdb
from dagster._utils.merger import merge_dicts
from .compute import OpExecutionContext
from .system import StepExecutionContext, TypeCheckContext

def _property_msg(prop_name: str, method_name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'The {prop_name} {method_name} is not set on the context when a solid is directly invoked.'

class UnboundOpExecutionContext(OpExecutionContext):
    """The ``context`` object available as the first argument to a solid's compute function when
    being invoked directly. Can also be used as a context manager.
    """

    def __init__(self, op_config: Any, resources_dict: Mapping[str, Any], resources_config: Mapping[str, Any], instance: Optional[DagsterInstance], partition_key: Optional[str], partition_key_range: Optional[PartitionKeyRange], mapping_key: Optional[str], assets_def: Optional[AssetsDefinition]):
        if False:
            for i in range(10):
                print('nop')
        from dagster._core.execution.api import ephemeral_instance_if_missing
        from dagster._core.execution.context_creation_job import initialize_console_manager
        self._op_config = op_config
        self._mapping_key = mapping_key
        self._exit_stack = ExitStack()
        self._instance = self._exit_stack.enter_context(ephemeral_instance_if_missing(instance))
        self._resources_config = resources_config
        self._resources_contain_cm = False
        self._resource_defs = wrap_resources_for_execution(resources_dict)
        self._resources = self._exit_stack.enter_context(build_resources(resources=self._resource_defs, instance=self._instance, resource_config=resources_config))
        self._resources_contain_cm = isinstance(self._resources, IContainsGenerator)
        self._log = initialize_console_manager(None)
        self._pdb: Optional[ForkedPdb] = None
        self._cm_scope_entered = False
        check.invariant(not (partition_key and partition_key_range), 'Must supply at most one of partition_key or partition_key_range')
        self._partition_key = partition_key
        self._partition_key_range = partition_key_range
        self._user_events: List[UserEvent] = []
        self._output_metadata: Dict[str, Any] = {}
        self._assets_def = check.opt_inst_param(assets_def, 'assets_def', AssetsDefinition)

    def __enter__(self):
        if False:
            return 10
        self._cm_scope_entered = True
        return self

    def __exit__(self, *exc):
        if False:
            while True:
                i = 10
        self._exit_stack.close()

    def __del__(self):
        if False:
            while True:
                i = 10
        self._exit_stack.close()

    @property
    def op_config(self) -> Any:
        if False:
            i = 10
            return i + 15
        return self._op_config

    @property
    def resource_keys(self) -> AbstractSet[str]:
        if False:
            i = 10
            return i + 15
        return self._resource_defs.keys()

    @property
    def resources(self) -> Resources:
        if False:
            i = 10
            return i + 15
        if self._resources_contain_cm and (not self._cm_scope_entered):
            raise DagsterInvariantViolationError('At least one provided resource is a generator, but attempting to access resources outside of context manager scope. You can use the following syntax to open a context manager: `with build_op_context(...) as context:`')
        return self._resources

    @property
    def dagster_run(self) -> DagsterRun:
        if False:
            print('Hello World!')
        raise DagsterInvalidPropertyError(_property_msg('pipeline_run', 'property'))

    @property
    def instance(self) -> DagsterInstance:
        if False:
            print('Hello World!')
        return self._instance

    @property
    def pdb(self) -> ForkedPdb:
        if False:
            print('Hello World!')
        'dagster.utils.forked_pdb.ForkedPdb: Gives access to pdb debugging from within the solid.\n\n        Example:\n        .. code-block:: python\n\n            @solid\n            def debug_solid(context):\n                context.pdb.set_trace()\n\n        '
        if self._pdb is None:
            self._pdb = ForkedPdb()
        return self._pdb

    @property
    def step_launcher(self) -> Optional[StepLauncher]:
        if False:
            return 10
        raise DagsterInvalidPropertyError(_property_msg('step_launcher', 'property'))

    @property
    def run_id(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'str: Hard-coded value to indicate that we are directly invoking solid.'
        return 'EPHEMERAL'

    @property
    def run_config(self) -> dict:
        if False:
            return 10
        raise DagsterInvalidPropertyError(_property_msg('run_config', 'property'))

    @property
    def job_def(self) -> JobDefinition:
        if False:
            i = 10
            return i + 15
        raise DagsterInvalidPropertyError(_property_msg('job_def', 'property'))

    @property
    def job_name(self) -> str:
        if False:
            while True:
                i = 10
        raise DagsterInvalidPropertyError(_property_msg('job_name', 'property'))

    @property
    def log(self) -> DagsterLogManager:
        if False:
            i = 10
            return i + 15
        'DagsterLogManager: A console manager constructed for this context.'
        return self._log

    @property
    def node_handle(self) -> NodeHandle:
        if False:
            i = 10
            return i + 15
        raise DagsterInvalidPropertyError(_property_msg('solid_handle', 'property'))

    @property
    def op(self) -> JobDefinition:
        if False:
            i = 10
            return i + 15
        raise DagsterInvalidPropertyError(_property_msg('op', 'property'))

    @property
    def solid(self) -> Node:
        if False:
            i = 10
            return i + 15
        raise DagsterInvalidPropertyError(_property_msg('solid', 'property'))

    @property
    def op_def(self) -> OpDefinition:
        if False:
            i = 10
            return i + 15
        raise DagsterInvalidPropertyError(_property_msg('op_def', 'property'))

    @property
    def assets_def(self) -> AssetsDefinition:
        if False:
            i = 10
            return i + 15
        raise DagsterInvalidPropertyError(_property_msg('assets_def', 'property'))

    @property
    def has_partition_key(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._partition_key is not None

    @property
    def partition_key(self) -> str:
        if False:
            print('Hello World!')
        if self._partition_key:
            return self._partition_key
        check.failed('Tried to access partition_key for a non-partitioned run')

    @property
    def partition_key_range(self) -> PartitionKeyRange:
        if False:
            return 10
        'The range of partition keys for the current run.\n\n        If run is for a single partition key, return a `PartitionKeyRange` with the same start and\n        end. Raises an error if the current run is not a partitioned run.\n        '
        if self._partition_key_range:
            return self._partition_key_range
        elif self._partition_key:
            return PartitionKeyRange(self._partition_key, self._partition_key)
        else:
            check.failed('Tried to access partition_key range for a non-partitioned run')

    def asset_partition_key_for_output(self, output_name: str='result') -> str:
        if False:
            while True:
                i = 10
        return self.partition_key

    def has_tag(self, key: str) -> bool:
        if False:
            i = 10
            return i + 15
        raise DagsterInvalidPropertyError(_property_msg('has_tag', 'method'))

    def get_tag(self, key: str) -> str:
        if False:
            i = 10
            return i + 15
        raise DagsterInvalidPropertyError(_property_msg('get_tag', 'method'))

    def get_step_execution_context(self) -> StepExecutionContext:
        if False:
            return 10
        raise DagsterInvalidPropertyError(_property_msg('get_step_execution_context', 'methods'))

    def bind(self, op_def: OpDefinition, pending_invocation: Optional[PendingNodeInvocation[OpDefinition]], assets_def: Optional[AssetsDefinition], config_from_args: Optional[Mapping[str, Any]], resources_from_args: Optional[Mapping[str, Any]]) -> 'BoundOpExecutionContext':
        if False:
            i = 10
            return i + 15
        from dagster._core.definitions.resource_invocation import resolve_bound_config
        if resources_from_args:
            if self._resource_defs:
                raise DagsterInvalidInvocationError('Cannot provide resources in both context and kwargs')
            resource_defs = wrap_resources_for_execution(resources_from_args)
            resources = self._exit_stack.enter_context(build_resources(resource_defs, self.instance))
        elif assets_def and assets_def.resource_defs:
            for key in sorted(list(assets_def.resource_defs.keys())):
                if key in self._resource_defs:
                    raise DagsterInvalidInvocationError(f"Error when invoking {assets_def!s} resource '{key}' provided on both the definition and invocation context. Please provide on only one or the other.")
            resource_defs = wrap_resources_for_execution({**self._resource_defs, **assets_def.resource_defs})
            resources = self._exit_stack.enter_context(build_resources(resource_defs, self.instance, self._resources_config))
        else:
            resources = self.resources
            resource_defs = self._resource_defs
        _validate_resource_requirements(resource_defs, op_def)
        if self.op_config and config_from_args:
            raise DagsterInvalidInvocationError('Cannot provide config in both context and kwargs')
        op_config = resolve_bound_config(config_from_args or self.op_config, op_def)
        return BoundOpExecutionContext(op_def=op_def, op_config=op_config, resources=resources, resources_config=self._resources_config, instance=self.instance, log_manager=self.log, pdb=self.pdb, tags=pending_invocation.tags if isinstance(pending_invocation, PendingNodeInvocation) else None, hook_defs=pending_invocation.hook_defs if isinstance(pending_invocation, PendingNodeInvocation) else None, alias=pending_invocation.given_alias if isinstance(pending_invocation, PendingNodeInvocation) else None, user_events=self._user_events, output_metadata=self._output_metadata, mapping_key=self._mapping_key, partition_key=self._partition_key, partition_key_range=self._partition_key_range, assets_def=assets_def)

    def get_events(self) -> Sequence[UserEvent]:
        if False:
            i = 10
            return i + 15
        'Retrieve the list of user-generated events that were logged via the context.\n\n        **Examples:**\n\n        .. code-block:: python\n\n            from dagster import op, build_op_context, AssetMaterialization, ExpectationResult\n\n            @op\n            def my_op(context):\n                ...\n\n            def test_my_op():\n                context = build_op_context()\n                my_op(context)\n                all_user_events = context.get_events()\n                materializations = [event for event in all_user_events if isinstance(event, AssetMaterialization)]\n                expectation_results = [event for event in all_user_events if isinstance(event, ExpectationResult)]\n                ...\n        '
        return self._user_events

    def get_output_metadata(self, output_name: str, mapping_key: Optional[str]=None) -> Optional[Mapping[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        'Retrieve metadata that was logged for an output and mapping_key, if it exists.\n\n        If metadata cannot be found for the particular output_name/mapping_key combination, None will be returned.\n\n        Args:\n            output_name (str): The name of the output to retrieve logged metadata for.\n            mapping_key (Optional[str]): The mapping key to retrieve metadata for (only applies when using dynamic outputs).\n\n        Returns:\n            Optional[Mapping[str, Any]]: The metadata values present for the output_name/mapping_key combination, if present.\n        '
        metadata = self._output_metadata.get(output_name)
        if mapping_key and metadata:
            return metadata.get(mapping_key)
        return metadata

    def get_mapping_key(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._mapping_key

def _validate_resource_requirements(resource_defs: Mapping[str, ResourceDefinition], op_def: OpDefinition) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Validate correctness of resources against required resource keys.'
    if cast(DecoratedOpFunction, op_def.compute_fn).has_context_arg():
        for requirement in op_def.get_resource_requirements():
            if not requirement.is_io_manager_requirement:
                ensure_requirements_satisfied(resource_defs, [requirement])

class BoundOpExecutionContext(OpExecutionContext):
    """The op execution context that is passed to the compute function during invocation.

    This context is bound to a specific op definition, for which the resources and config have
    been validated.
    """
    _op_def: OpDefinition
    _op_config: Any
    _resources: 'Resources'
    _resources_config: Mapping[str, Any]
    _instance: DagsterInstance
    _log_manager: DagsterLogManager
    _pdb: Optional[ForkedPdb]
    _tags: Mapping[str, str]
    _hook_defs: Optional[AbstractSet[HookDefinition]]
    _alias: str
    _user_events: List[UserEvent]
    _seen_outputs: Dict[str, Union[str, Set[str]]]
    _output_metadata: Dict[str, Any]
    _mapping_key: Optional[str]
    _partition_key: Optional[str]
    _partition_key_range: Optional[PartitionKeyRange]
    _assets_def: Optional[AssetsDefinition]

    def __init__(self, op_def: OpDefinition, op_config: Any, resources: 'Resources', resources_config: Mapping[str, Any], instance: DagsterInstance, log_manager: DagsterLogManager, pdb: Optional[ForkedPdb], tags: Optional[Mapping[str, str]], hook_defs: Optional[AbstractSet[HookDefinition]], alias: Optional[str], user_events: List[UserEvent], output_metadata: Dict[str, Any], mapping_key: Optional[str], partition_key: Optional[str], partition_key_range: Optional[PartitionKeyRange], assets_def: Optional[AssetsDefinition]):
        if False:
            i = 10
            return i + 15
        self._op_def = op_def
        self._op_config = op_config
        self._resources = resources
        self._instance = instance
        self._log = log_manager
        self._pdb = pdb
        self._tags = merge_dicts(self._op_def.tags, tags) if tags else self._op_def.tags
        self._hook_defs = hook_defs
        self._alias = alias if alias else self._op_def.name
        self._resources_config = resources_config
        self._user_events = user_events
        self._seen_outputs = {}
        self._output_metadata = output_metadata
        self._mapping_key = mapping_key
        self._partition_key = partition_key
        self._partition_key_range = partition_key_range
        self._assets_def = assets_def
        self._requires_typed_event_stream = False
        self._typed_event_stream_error_message = None

    @property
    def op_config(self) -> Any:
        if False:
            while True:
                i = 10
        return self._op_config

    @property
    def resources(self) -> Resources:
        if False:
            i = 10
            return i + 15
        return self._resources

    @property
    def dagster_run(self) -> DagsterRun:
        if False:
            print('Hello World!')
        raise DagsterInvalidPropertyError(_property_msg('pipeline_run', 'property'))

    @property
    def instance(self) -> DagsterInstance:
        if False:
            print('Hello World!')
        return self._instance

    @property
    def pdb(self) -> ForkedPdb:
        if False:
            return 10
        'dagster.utils.forked_pdb.ForkedPdb: Gives access to pdb debugging from within the solid.\n\n        Example:\n        .. code-block:: python\n\n            @solid\n            def debug_solid(context):\n                context.pdb.set_trace()\n\n        '
        if self._pdb is None:
            self._pdb = ForkedPdb()
        return self._pdb

    @property
    def step_launcher(self) -> Optional[StepLauncher]:
        if False:
            print('Hello World!')
        raise DagsterInvalidPropertyError(_property_msg('step_launcher', 'property'))

    @property
    def run_id(self) -> str:
        if False:
            return 10
        'str: Hard-coded value to indicate that we are directly invoking solid.'
        return 'EPHEMERAL'

    @property
    def run_config(self) -> Mapping[str, object]:
        if False:
            while True:
                i = 10
        run_config: Dict[str, object] = {}
        if self._op_config:
            run_config['ops'] = {self._op_def.name: {'config': self._op_config}}
        run_config['resources'] = self._resources_config
        return run_config

    @property
    def job_def(self) -> JobDefinition:
        if False:
            return 10
        raise DagsterInvalidPropertyError(_property_msg('job_def', 'property'))

    @property
    def job_name(self) -> str:
        if False:
            return 10
        raise DagsterInvalidPropertyError(_property_msg('job_name', 'property'))

    @property
    def log(self) -> DagsterLogManager:
        if False:
            for i in range(10):
                print('nop')
        'DagsterLogManager: A console manager constructed for this context.'
        return self._log

    @property
    def node_handle(self) -> NodeHandle:
        if False:
            i = 10
            return i + 15
        raise DagsterInvalidPropertyError(_property_msg('node_handle', 'property'))

    @property
    def op(self) -> Node:
        if False:
            i = 10
            return i + 15
        raise DagsterInvalidPropertyError(_property_msg('op', 'property'))

    @property
    def op_def(self) -> OpDefinition:
        if False:
            while True:
                i = 10
        return self._op_def

    @property
    def has_assets_def(self) -> bool:
        if False:
            while True:
                i = 10
        return self._assets_def is not None

    @property
    def assets_def(self) -> AssetsDefinition:
        if False:
            return 10
        if self._assets_def is None:
            raise DagsterInvalidPropertyError(f'Op {self.op_def.name} does not have an assets definition.')
        return self._assets_def

    @property
    def has_partition_key(self) -> bool:
        if False:
            return 10
        return self._partition_key is not None

    def has_tag(self, key: str) -> bool:
        if False:
            while True:
                i = 10
        return key in self._tags

    def get_tag(self, key: str) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._tags.get(key)

    @property
    def alias(self) -> str:
        if False:
            while True:
                i = 10
        return self._alias

    def get_step_execution_context(self) -> StepExecutionContext:
        if False:
            return 10
        raise DagsterInvalidPropertyError(_property_msg('get_step_execution_context', 'methods'))

    def for_type(self, dagster_type: DagsterType) -> TypeCheckContext:
        if False:
            for i in range(10):
                print('nop')
        resources = cast(NamedTuple, self.resources)
        return TypeCheckContext(self.run_id, self.log, ScopedResourcesBuilder(resources._asdict()), dagster_type)

    def get_mapping_key(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._mapping_key

    def describe_op(self) -> str:
        if False:
            while True:
                i = 10
        if isinstance(self.op_def, OpDefinition):
            return f'op "{self.op_def.name}"'
        return f'solid "{self.op_def.name}"'

    def log_event(self, event: UserEvent) -> None:
        if False:
            while True:
                i = 10
        check.inst_param(event, 'event', (AssetMaterialization, AssetObservation, ExpectationResult))
        self._user_events.append(event)

    def observe_output(self, output_name: str, mapping_key: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        if mapping_key:
            if output_name not in self._seen_outputs:
                self._seen_outputs[output_name] = set()
            cast(Set[str], self._seen_outputs[output_name]).add(mapping_key)
        else:
            self._seen_outputs[output_name] = 'seen'

    def has_seen_output(self, output_name: str, mapping_key: Optional[str]=None) -> bool:
        if False:
            i = 10
            return i + 15
        if mapping_key:
            return output_name in self._seen_outputs and mapping_key in self._seen_outputs[output_name]
        return output_name in self._seen_outputs

    @property
    def partition_key(self) -> str:
        if False:
            return 10
        if self._partition_key is not None:
            return self._partition_key
        check.failed('Tried to access partition_key for a non-partitioned asset')

    @property
    def partition_key_range(self) -> PartitionKeyRange:
        if False:
            for i in range(10):
                print('nop')
        'The range of partition keys for the current run.\n\n        If run is for a single partition key, return a `PartitionKeyRange` with the same start and\n        end. Raises an error if the current run is not a partitioned run.\n        '
        if self._partition_key_range:
            return self._partition_key_range
        elif self._partition_key:
            return PartitionKeyRange(self._partition_key, self._partition_key)
        else:
            check.failed('Tried to access partition_key range for a non-partitioned run')

    def asset_partition_key_for_output(self, output_name: str='result') -> str:
        if False:
            print('Hello World!')
        return self.partition_key

    def asset_partitions_time_window_for_output(self, output_name: str='result') -> TimeWindow:
        if False:
            return 10
        partitions_def = self.assets_def.partitions_def
        if partitions_def is None:
            check.failed('Tried to access partition_key for a non-partitioned asset')
        if not has_one_dimension_time_window_partitioning(partitions_def=partitions_def):
            raise DagsterInvariantViolationError(f'Expected a TimeWindowPartitionsDefinition or MultiPartitionsDefinition with a single time dimension, but instead found {type(partitions_def)}')
        return cast(Union[MultiPartitionsDefinition, TimeWindowPartitionsDefinition], partitions_def).time_window_for_partition_key(self.partition_key)

    def add_output_metadata(self, metadata: Mapping[str, Any], output_name: Optional[str]=None, mapping_key: Optional[str]=None) -> None:
        if False:
            return 10
        'Add metadata to one of the outputs of an op.\n\n        This can only be used once per output in the body of an op. Using this method with the same output_name more than once within an op will result in an error.\n\n        Args:\n            metadata (Mapping[str, Any]): The metadata to attach to the output\n            output_name (Optional[str]): The name of the output to attach metadata to. If there is only one output on the op, then this argument does not need to be provided. The metadata will automatically be attached to the only output.\n\n        **Examples:**\n\n        .. code-block:: python\n\n            from dagster import Out, op\n            from typing import Tuple\n\n            @op\n            def add_metadata(context):\n                context.add_output_metadata({"foo", "bar"})\n                return 5 # Since the default output is called "result", metadata will be attached to the output "result".\n\n            @op(out={"a": Out(), "b": Out()})\n            def add_metadata_two_outputs(context) -> Tuple[str, int]:\n                context.add_output_metadata({"foo": "bar"}, output_name="b")\n                context.add_output_metadata({"baz": "bat"}, output_name="a")\n\n                return ("dog", 5)\n\n        '
        metadata = check.mapping_param(metadata, 'metadata', key_type=str)
        output_name = check.opt_str_param(output_name, 'output_name')
        mapping_key = check.opt_str_param(mapping_key, 'mapping_key')
        if output_name is None and len(self.op_def.output_defs) == 1:
            output_def = self.op_def.output_defs[0]
            output_name = output_def.name
        elif output_name is None:
            raise DagsterInvariantViolationError('Attempted to log metadata without providing output_name, but multiple outputs exist. Please provide an output_name to the invocation of `context.add_output_metadata`.')
        else:
            output_def = self.op_def.output_def_named(output_name)
        if self.has_seen_output(output_name, mapping_key):
            output_desc = f"output '{output_def.name}'" if not mapping_key else f"output '{output_def.name}' with mapping_key '{mapping_key}'"
            raise DagsterInvariantViolationError(f"In {self.op_def.node_type_str} '{self.op_def.name}', attempted to log output metadata for {output_desc} which has already been yielded. Metadata must be logged before the output is yielded.")
        if output_def.is_dynamic and (not mapping_key):
            raise DagsterInvariantViolationError(f"In {self.op_def.node_type_str} '{self.op_def.name}', attempted to log metadata for dynamic output '{output_def.name}' without providing a mapping key. When logging metadata for a dynamic output, it is necessary to provide a mapping key.")
        output_name = output_def.name
        if output_name in self._output_metadata:
            if not mapping_key or mapping_key in self._output_metadata[output_name]:
                raise DagsterInvariantViolationError(f"In {self.op_def.node_type_str} '{self.op_def.name}', attempted to log metadata for output '{output_name}' more than once.")
        if mapping_key:
            if output_name not in self._output_metadata:
                self._output_metadata[output_name] = {}
            self._output_metadata[output_name][mapping_key] = metadata
        else:
            self._output_metadata[output_name] = metadata

    @property
    def requires_typed_event_stream(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._requires_typed_event_stream

    @property
    def typed_event_stream_error_message(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._typed_event_stream_error_message

    def set_requires_typed_event_stream(self, *, error_message: Optional[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._requires_typed_event_stream = True
        self._typed_event_stream_error_message = error_message

def build_op_context(resources: Optional[Mapping[str, Any]]=None, op_config: Any=None, resources_config: Optional[Mapping[str, Any]]=None, instance: Optional[DagsterInstance]=None, config: Any=None, partition_key: Optional[str]=None, partition_key_range: Optional[PartitionKeyRange]=None, mapping_key: Optional[str]=None, _assets_def: Optional[AssetsDefinition]=None) -> UnboundOpExecutionContext:
    if False:
        i = 10
        return i + 15
    'Builds op execution context from provided parameters.\n\n    ``build_op_context`` can be used as either a function or context manager. If there is a\n    provided resource that is a context manager, then ``build_op_context`` must be used as a\n    context manager. This function can be used to provide the context argument when directly\n    invoking a op.\n\n    Args:\n        resources (Optional[Dict[str, Any]]): The resources to provide to the context. These can be\n            either values or resource definitions.\n        op_config (Optional[Mapping[str, Any]]): The config to provide to the op.\n        resources_config (Optional[Mapping[str, Any]]): The config to provide to the resources.\n        instance (Optional[DagsterInstance]): The dagster instance configured for the context.\n            Defaults to DagsterInstance.ephemeral().\n        mapping_key (Optional[str]): A key representing the mapping key from an upstream dynamic\n            output. Can be accessed using ``context.get_mapping_key()``.\n        partition_key (Optional[str]): String value representing partition key to execute with.\n        partition_key_range (Optional[PartitionKeyRange]): Partition key range to execute with.\n        _assets_def (Optional[AssetsDefinition]): Internal argument that populates the op\'s assets\n            definition, not meant to be populated by users.\n\n    Examples:\n        .. code-block:: python\n\n            context = build_op_context()\n            op_to_invoke(context)\n\n            with build_op_context(resources={"foo": context_manager_resource}) as context:\n                op_to_invoke(context)\n    '
    if op_config and config:
        raise DagsterInvalidInvocationError('Attempted to invoke ``build_op_context`` with both ``op_config``, and its legacy version, ``config``. Please provide one or the other.')
    op_config = op_config if op_config else config
    return UnboundOpExecutionContext(resources_dict=check.opt_mapping_param(resources, 'resources', key_type=str), resources_config=check.opt_mapping_param(resources_config, 'resources_config', key_type=str), op_config=op_config, instance=check.opt_inst_param(instance, 'instance', DagsterInstance), partition_key=check.opt_str_param(partition_key, 'partition_key'), partition_key_range=check.opt_inst_param(partition_key_range, 'partition_key_range', PartitionKeyRange), mapping_key=check.opt_str_param(mapping_key, 'mapping_key'), assets_def=check.opt_inst_param(_assets_def, '_assets_def', AssetsDefinition))

def build_asset_context(resources: Optional[Mapping[str, Any]]=None, resources_config: Optional[Mapping[str, Any]]=None, asset_config: Optional[Mapping[str, Any]]=None, instance: Optional[DagsterInstance]=None, partition_key: Optional[str]=None, partition_key_range: Optional[PartitionKeyRange]=None):
    if False:
        while True:
            i = 10
    'Builds asset execution context from provided parameters.\n\n    ``build_asset_context`` can be used as either a function or context manager. If there is a\n    provided resource that is a context manager, then ``build_asset_context`` must be used as a\n    context manager. This function can be used to provide the context argument when directly\n    invoking an asset.\n\n    Args:\n        resources (Optional[Dict[str, Any]]): The resources to provide to the context. These can be\n            either values or resource definitions.\n        resources_config (Optional[Mapping[str, Any]]): The config to provide to the resources.\n        asset_config (Optional[Mapping[str, Any]]): The config to provide to the asset.\n        instance (Optional[DagsterInstance]): The dagster instance configured for the context.\n            Defaults to DagsterInstance.ephemeral().\n        partition_key (Optional[str]): String value representing partition key to execute with.\n        partition_key_range (Optional[PartitionKeyRange]): Partition key range to execute with.\n\n    Examples:\n        .. code-block:: python\n\n            context = build_asset_context()\n            asset_to_invoke(context)\n\n            with build_asset_context(resources={"foo": context_manager_resource}) as context:\n                asset_to_invoke(context)\n    '
    return build_op_context(op_config=asset_config, resources=resources, resources_config=resources_config, partition_key=partition_key, partition_key_range=partition_key_range, instance=instance)