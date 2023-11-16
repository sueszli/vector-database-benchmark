from typing import Callable, Iterator, Mapping, NamedTuple, NoReturn, cast
from typing_extensions import TypeAlias
import dagster._check as check
from dagster._config import EvaluateValueResult, process_config
from dagster._core.definitions.asset_layer import AssetLayer
from dagster._core.definitions.dependency import GraphNode, Node, NodeHandle, OpNode
from dagster._core.definitions.graph_definition import GraphDefinition, SubselectedGraphDefinition
from dagster._core.definitions.job_definition import JobDefinition
from dagster._core.definitions.resource_definition import ResourceDefinition
from dagster._core.definitions.run_config import define_node_shape
from dagster._core.errors import DagsterConfigMappingFunctionError, DagsterInvalidConfigError, user_code_error_boundary
from dagster._core.system_config.objects import OpConfig
from dagster._utils.merger import merge_dicts
RawNodeConfig: TypeAlias = Mapping[str, object]

class OpConfigEntry(NamedTuple('_SolidConfigEntry', [('handle', NodeHandle), ('solid_config', OpConfig)])):

    def __new__(cls, handle: NodeHandle, op_config: OpConfig):
        if False:
            i = 10
            return i + 15
        return super(OpConfigEntry, cls).__new__(cls, check.inst_param(handle, 'handle', NodeHandle), check.inst_param(op_config, 'solid_config', OpConfig))
_ROOT_HANDLE = NodeHandle('root', None)

class DescentStack(NamedTuple('_DescentStack', [('job_def', JobDefinition), ('handle', NodeHandle)])):

    def __new__(cls, job_def: JobDefinition, handle: NodeHandle):
        if False:
            print('Hello World!')
        return super(DescentStack, cls).__new__(cls, job_def=check.inst_param(job_def, 'job_def', JobDefinition), handle=check.inst_param(handle, 'handle', NodeHandle))

    @property
    def current_container(self) -> GraphDefinition:
        if False:
            print('Hello World!')
        if self.handle == _ROOT_HANDLE:
            return self.job_def.graph
        else:
            assert isinstance(self.current_node, GraphNode)
            return self.current_node.definition

    @property
    def current_node(self) -> Node:
        if False:
            print('Hello World!')
        assert self.handle is not None
        return self.job_def.get_node(self.handle)

    @property
    def current_handle_str(self) -> str:
        if False:
            print('Hello World!')
        return check.not_none(self.handle).to_string()

    def descend(self, node: Node) -> 'DescentStack':
        if False:
            while True:
                i = 10
        parent = self.handle if self.handle != _ROOT_HANDLE else None
        return self._replace(handle=NodeHandle(node.name, parent=parent))

def composite_descent(job_def: JobDefinition, ops_config: Mapping[str, RawNodeConfig], resource_defs: Mapping[str, ResourceDefinition]) -> Mapping[str, OpConfig]:
    if False:
        for i in range(10):
            print('nop')
    'This function is responsible for constructing the dictionary of OpConfig (indexed by handle)\n    that will be passed into the ResolvedRunConfig. Critically this is the codepath that manages\n    config mapping, where the runtime calls into user-defined config mapping functions to produce\n    config for child nodes of graphs.\n\n    Args:\n        job_def (JobDefinition): JobDefinition\n        ops_config (dict): Configuration for the ops in the pipeline. The "ops" entry\n            of the run_config. Assumed to have already been validated.\n\n    Returns:\n        Dict[str, OpConfig]: A dictionary mapping string representations of NodeHandles to\n            OpConfig objects. It includes an entry for ops at every level of the\n            composite tree - i.e. not just leaf ops, but composite ops as well\n    '
    check.inst_param(job_def, 'job_def', JobDefinition)
    check.dict_param(ops_config, 'ops_config')
    check.dict_param(resource_defs, 'resource_defs', key_type=str, value_type=ResourceDefinition)
    if job_def.graph.has_config_mapping:
        ops_config = _apply_top_level_config_mapping(job_def, ops_config, resource_defs)
    return {handle.to_string(): op_config for (handle, op_config) in _composite_descent(parent_stack=DescentStack(job_def, _ROOT_HANDLE), ops_config_dict=ops_config, resource_defs=resource_defs, asset_layer=job_def.asset_layer)}

def _composite_descent(parent_stack: DescentStack, ops_config_dict: Mapping[str, RawNodeConfig], resource_defs: Mapping[str, ResourceDefinition], asset_layer: AssetLayer) -> Iterator[OpConfigEntry]:
    if False:
        while True:
            i = 10
    'The core implementation of composite_descent. This yields a stream of OpConfigEntry. This is\n    used by composite_descent to construct a dictionary.\n\n    It descends over the entire node hierarchy, constructing an entry for every handle. If it\n    encounters a graph instance with a config mapping, it will invoke that config mapping fn,\n    producing the config that is necessary to configure the child nodes.\n\n    This process unrolls recursively as you descend down the tree.\n    '
    for node in parent_stack.current_container.nodes:
        current_stack = parent_stack.descend(node)
        current_handle = current_stack.handle
        current_op_config = ops_config_dict.get(node.name, {})
        if isinstance(node, OpNode):
            config_mapped_node_config = node.definition.apply_config_mapping({'config': current_op_config.get('config')})
            if not config_mapped_node_config.success:
                raise DagsterInvalidConfigError(f'Error in config for {node.describe_node()}'.format(node.name), config_mapped_node_config.errors, config_mapped_node_config)
            complete_config_object = merge_dicts(current_op_config, config_mapped_node_config.value)
            yield OpConfigEntry(current_handle, OpConfig.from_dict(complete_config_object))
            continue
        elif isinstance(node, GraphNode):
            yield OpConfigEntry(current_handle, OpConfig.from_dict({'inputs': current_op_config.get('inputs'), 'outputs': current_op_config.get('outputs')}))
            mapped_nodes_config = _apply_config_mapping(node, current_stack, current_op_config, resource_defs, asset_layer) if node.definition.has_config_mapping else cast(Mapping[str, RawNodeConfig], current_op_config.get('ops', {}))
            yield from _composite_descent(current_stack, mapped_nodes_config, resource_defs, asset_layer)
        else:
            check.failed(f'Unexpected node type {type(node)}')

def _apply_top_level_config_mapping(job_def: JobDefinition, outer_config: Mapping[str, Mapping[str, object]], resource_defs: Mapping[str, ResourceDefinition]) -> Mapping[str, RawNodeConfig]:
    if False:
        while True:
            i = 10
    graph_def = job_def.graph
    config_mapping = graph_def.config_mapping
    if config_mapping is None:
        return outer_config
    else:
        mapped_config_evr = graph_def.apply_config_mapping(outer_config)
        if not mapped_config_evr.success:
            raise DagsterInvalidConfigError(f'Error in config for graph {graph_def.name}', mapped_config_evr.errors, outer_config)
        with user_code_error_boundary(DagsterConfigMappingFunctionError, _get_top_level_error_lambda(job_def)):
            mapped_graph_config = config_mapping.resolve_from_validated_config(mapped_config_evr.value.get('config', {}))
        type_to_evaluate_against = define_node_shape(nodes=graph_def.nodes, ignored_nodes=None, dependency_structure=graph_def.dependency_structure, resource_defs=resource_defs, asset_layer=job_def.asset_layer, node_input_source_assets=graph_def.node_input_source_assets)
        evr = process_config(type_to_evaluate_against, mapped_graph_config)
        if not evr.success:
            raise_top_level_config_error(job_def, mapped_graph_config, evr)
        return evr.value

def _apply_config_mapping(graph_node: GraphNode, current_stack: DescentStack, current_node_config: RawNodeConfig, resource_defs: Mapping[str, ResourceDefinition], asset_layer: AssetLayer) -> Mapping[str, RawNodeConfig]:
    if False:
        i = 10
        return i + 15
    graph_def = graph_node.definition
    config_mapped_node_config = graph_def.apply_config_mapping(current_node_config)
    if not config_mapped_node_config.success:
        raise DagsterInvalidConfigError(f'Error in config for graph {graph_node.name}', config_mapped_node_config.errors, config_mapped_node_config)
    with user_code_error_boundary(DagsterConfigMappingFunctionError, _get_error_lambda(current_stack)):
        config_mapping = check.not_none(graph_def.config_mapping)
        mapped_ops_config = config_mapping.resolve_from_validated_config(config_mapped_node_config.value.get('config', {}))
    ignored_nodes = graph_def.get_top_level_omitted_nodes() if isinstance(graph_def, SubselectedGraphDefinition) else None
    type_to_evaluate_against = define_node_shape(nodes=graph_def.nodes, ignored_nodes=ignored_nodes, dependency_structure=graph_def.dependency_structure, parent_handle=current_stack.handle, resource_defs=resource_defs, asset_layer=asset_layer, node_input_source_assets=graph_def.node_input_source_assets)
    evr = process_config(type_to_evaluate_against, mapped_ops_config)
    if not evr.success:
        raise_composite_descent_config_error(current_stack, mapped_ops_config, evr)
    return evr.value

def _get_error_lambda(current_stack: DescentStack) -> Callable[[], str]:
    if False:
        return 10
    return lambda : 'The config mapping function on {described_node} in {described_target} has thrown an unexpected error during its execution. The definition is instantiated at stack "{stack_str}".'.format(described_node=current_stack.current_node.describe_node(), described_target=current_stack.job_def.describe_target(), stack_str=':'.join(current_stack.handle.path))

def _get_top_level_error_lambda(job_def: JobDefinition) -> Callable[[], str]:
    if False:
        while True:
            i = 10
    return lambda : f'The config mapping function on top-level graph {job_def.graph.name} in job {job_def.name} has thrown an unexpected error during its execution.'

def raise_top_level_config_error(job_def: JobDefinition, failed_config_value: object, evr: EvaluateValueResult) -> NoReturn:
    if False:
        return 10
    message = f"In job '{job_def.name}', top level graph '{job_def.graph.name}' has a configuration error."
    raise DagsterInvalidConfigError(message, evr.errors, failed_config_value)

def raise_composite_descent_config_error(descent_stack: DescentStack, failed_config_value: object, evr: EvaluateValueResult) -> NoReturn:
    if False:
        i = 10
        return i + 15
    check.inst_param(descent_stack, 'descent_stack', DescentStack)
    check.inst_param(evr, 'evr', EvaluateValueResult)
    node = descent_stack.current_node
    message = 'In job {job_name} at stack {stack}: \n'.format(job_name=descent_stack.job_def.name, stack=':'.join(descent_stack.handle.path))
    message += f'Op "{node.name}" with definition "{node.definition.name}" has a configuration error. It has produced config a via its config_fn that fails to pass validation in the ops that it contains. This indicates an error in the config mapping function itself. It must produce correct config for its constituent ops in all cases. The correct resolution is to fix the mapping function. Details on the error (and the paths on this error are relative to config mapping function "root", not the entire document): '
    raise DagsterInvalidConfigError(message, evr.errors, failed_config_value)