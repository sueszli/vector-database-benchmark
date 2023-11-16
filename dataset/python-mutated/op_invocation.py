import inspect
from typing import TYPE_CHECKING, Any, Dict, Mapping, NamedTuple, Set, Tuple, TypeVar, Union, cast
import dagster._check as check
from dagster._core.decorator_utils import get_function_params
from dagster._core.errors import DagsterInvalidInvocationError, DagsterInvariantViolationError, DagsterTypeCheckDidNotPass
from .events import AssetKey, AssetMaterialization, AssetObservation, DynamicOutput, ExpectationResult, Output
from .output import DynamicOutputDefinition, OutputDefinition
from .result import MaterializeResult
if TYPE_CHECKING:
    from ..execution.context.invocation import BoundOpExecutionContext
    from .assets import AssetsDefinition
    from .composition import PendingNodeInvocation
    from .decorators.op_decorator import DecoratedOpFunction
    from .op_definition import OpDefinition
T = TypeVar('T')

class SeparatedArgsKwargs(NamedTuple):
    input_args: Tuple[Any, ...]
    input_kwargs: Dict[str, Any]
    resources_by_param_name: Dict[str, Any]
    config_arg: Any

def _separate_args_and_kwargs(compute_fn: 'DecoratedOpFunction', args: Tuple[Any, ...], kwargs: Dict[str, Any], resource_arg_mapping: Dict[str, Any]) -> SeparatedArgsKwargs:
    if False:
        print('Hello World!')
    'Given a decorated compute function, a set of args and kwargs, and set of resource param names,\n    separates the set of resource inputs from op/asset inputs returns a tuple of the categorized\n    args and kwargs.\n\n    We use the remaining args and kwargs to cleanly invoke the compute function, and we use the\n    extracted resource inputs to populate the execution context.\n    '
    resources_from_args_and_kwargs = {}
    params = get_function_params(compute_fn.decorated_fn)
    adjusted_args = []
    params_without_context = params[1:] if compute_fn.has_context_arg() else params
    config_arg = kwargs.get('config')
    for (i, arg) in enumerate(args):
        param = params_without_context[i] if i < len(params_without_context) else None
        if param and param.kind != inspect.Parameter.KEYWORD_ONLY:
            if param.name in resource_arg_mapping:
                resources_from_args_and_kwargs[param.name] = arg
                continue
            if param.name == 'config':
                config_arg = arg
                continue
        adjusted_args.append(arg)
    for resource_arg in resource_arg_mapping:
        if resource_arg in kwargs:
            resources_from_args_and_kwargs[resource_arg] = kwargs[resource_arg]
    adjusted_kwargs = {k: v for (k, v) in kwargs.items() if k not in resources_from_args_and_kwargs and k != 'config'}
    return SeparatedArgsKwargs(input_args=tuple(adjusted_args), input_kwargs=adjusted_kwargs, resources_by_param_name=resources_from_args_and_kwargs, config_arg=config_arg)

def direct_invocation_result(def_or_invocation: Union['OpDefinition', 'PendingNodeInvocation[OpDefinition]', 'AssetsDefinition'], *args, **kwargs) -> Any:
    if False:
        return 10
    from dagster._config.pythonic_config import Config
    from dagster._core.execution.context.invocation import UnboundOpExecutionContext, build_op_context
    from ..execution.plan.compute_generator import invoke_compute_fn
    from .assets import AssetsDefinition
    from .composition import PendingNodeInvocation
    from .decorators.op_decorator import DecoratedOpFunction
    from .op_definition import OpDefinition
    if isinstance(def_or_invocation, OpDefinition):
        op_def = def_or_invocation
        pending_invocation = None
        assets_def = None
    elif isinstance(def_or_invocation, AssetsDefinition):
        assets_def = def_or_invocation
        op_def = assets_def.op
        pending_invocation = None
    elif isinstance(def_or_invocation, PendingNodeInvocation):
        pending_invocation = def_or_invocation
        op_def = def_or_invocation.node_def
        assets_def = None
    else:
        check.failed(f'unexpected direct invocation target {def_or_invocation}')
    compute_fn = op_def.compute_fn
    if not isinstance(compute_fn, DecoratedOpFunction):
        raise DagsterInvalidInvocationError('Attempted to directly invoke an op/asset that was not constructed using the `@op` or `@asset` decorator. Only decorated functions can be directly invoked.')
    context = None
    if compute_fn.has_context_arg():
        if len(args) + len(kwargs) == 0:
            raise DagsterInvalidInvocationError(f"Decorated function '{compute_fn.name}' has context argument, but no context was provided when invoking.")
        if len(args) > 0:
            if args[0] is not None and (not isinstance(args[0], UnboundOpExecutionContext)):
                raise DagsterInvalidInvocationError(f"Decorated function '{compute_fn.name}' has context argument, but no context was provided when invoking.")
            context = cast(UnboundOpExecutionContext, args[0])
            args = args[1:]
        else:
            context_param_name = get_function_params(compute_fn.decorated_fn)[0].name
            if context_param_name not in kwargs:
                raise DagsterInvalidInvocationError(f"Decorated function '{compute_fn.name}' has context argument '{context_param_name}', but no value for '{context_param_name}' was found when invoking. Provided kwargs: {kwargs}")
            context = cast(UnboundOpExecutionContext, kwargs[context_param_name])
            kwargs = {kwarg: val for (kwarg, val) in kwargs.items() if not kwarg == context_param_name}
    elif len(args) > 0 and isinstance(args[0], UnboundOpExecutionContext):
        context = cast(UnboundOpExecutionContext, args[0])
        args = args[1:]
    resource_arg_mapping = {arg.name: arg.name for arg in compute_fn.get_resource_args()}
    extracted = _separate_args_and_kwargs(compute_fn, args, kwargs, resource_arg_mapping)
    input_args = extracted.input_args
    input_kwargs = extracted.input_kwargs
    resources_by_param_name = extracted.resources_by_param_name
    config_input = extracted.config_arg
    bound_context = (context or build_op_context()).bind(op_def=op_def, pending_invocation=pending_invocation, assets_def=assets_def, resources_from_args=resources_by_param_name, config_from_args=config_input._convert_to_config_dictionary() if isinstance(config_input, Config) else config_input)
    input_dict = _resolve_inputs(op_def, input_args, input_kwargs, bound_context)
    result = invoke_compute_fn(fn=compute_fn.decorated_fn, context=bound_context, kwargs=input_dict, context_arg_provided=compute_fn.has_context_arg(), config_arg_cls=compute_fn.get_config_arg().annotation if compute_fn.has_config_arg() else None, resource_args=resource_arg_mapping)
    return _type_check_output_wrapper(op_def, result, bound_context)

def _resolve_inputs(op_def: 'OpDefinition', args, kwargs, context: 'BoundOpExecutionContext') -> Mapping[str, Any]:
    if False:
        return 10
    from dagster._core.execution.plan.execute_step import do_type_check
    nothing_input_defs = [input_def for input_def in op_def.input_defs if input_def.dagster_type.is_nothing]
    for input_def in nothing_input_defs:
        if input_def.name in kwargs:
            node_label = op_def.node_type_str
            raise DagsterInvalidInvocationError(f"Attempted to provide value for nothing input '{input_def.name}'. Nothing dependencies are ignored when directly invoking {node_label}s.")
    input_defs_by_name = {input_def.name: input_def for input_def in op_def.input_defs if not input_def.dagster_type.is_nothing}
    if len(input_defs_by_name) < len(args) + len(kwargs):
        if len(nothing_input_defs) > 0:
            suggestion = 'This may be because you attempted to provide a value for a nothing dependency. Nothing dependencies are ignored when directly invoking ops.'
        else:
            suggestion = 'This may be because an argument was provided for the context parameter, but no context parameter was defined for the op.'
        node_label = op_def.node_type_str
        raise DagsterInvalidInvocationError(f"Too many input arguments were provided for {node_label} '{context.alias}'. {suggestion}")
    positional_inputs = cast('DecoratedOpFunction', op_def.compute_fn).positional_inputs()
    if len(args) > len(positional_inputs):
        raise DagsterInvalidInvocationError(f"{op_def.node_type_str} '{op_def.name}' has {len(positional_inputs)} positional inputs, but {len(args)} positional inputs were provided.")
    input_dict = {}
    for (position, value) in enumerate(args):
        input_name = positional_inputs[position]
        input_dict[input_name] = value
        if input_name in kwargs:
            raise DagsterInvalidInvocationError(f"{op_def.node_type_str} {op_def.name} got multiple values for argument '{input_name}'")
    for input_name in positional_inputs[len(args):]:
        input_def = input_defs_by_name[input_name]
        if input_name in kwargs:
            input_dict[input_name] = kwargs[input_name]
        elif input_def.has_default_value:
            input_dict[input_name] = input_def.default_value
        else:
            raise DagsterInvalidInvocationError(f'No value provided for required input "{input_name}".')
    unassigned_kwargs = {k: v for (k, v) in kwargs.items() if k not in input_dict}
    if unassigned_kwargs and cast('DecoratedOpFunction', op_def.compute_fn).has_var_kwargs():
        for (k, v) in unassigned_kwargs.items():
            input_dict[k] = v
    op_label = context.describe_op()
    for (input_name, val) in input_dict.items():
        input_def = input_defs_by_name[input_name]
        dagster_type = input_def.dagster_type
        type_check = do_type_check(context.for_type(dagster_type), dagster_type, val)
        if not type_check.success:
            raise DagsterTypeCheckDidNotPass(description=f'Type check failed for {op_label} input "{input_def.name}" - expected type "{dagster_type.display_name}". Description: {type_check.description}', metadata=type_check.metadata, dagster_type=dagster_type)
    return input_dict

def _key_for_result(result: MaterializeResult, context: 'BoundOpExecutionContext') -> AssetKey:
    if False:
        print('Hello World!')
    if result.asset_key:
        return result.asset_key
    if len(context.assets_def.keys) == 1:
        return next(iter(context.assets_def.keys))
    raise DagsterInvariantViolationError(f'MaterializeResult did not include asset_key and it can not be inferred. Specify which asset_key, options are: {context.assets_def.keys}')

def _output_name_for_result_obj(event: MaterializeResult, context: 'BoundOpExecutionContext'):
    if False:
        i = 10
        return i + 15
    asset_key = _key_for_result(event, context)
    return context.assets_def.get_output_name_for_asset_key(asset_key)

def _handle_gen_event(event: T, op_def: 'OpDefinition', context: 'BoundOpExecutionContext', output_defs: Mapping[str, OutputDefinition], outputs_seen: Set[str]) -> T:
    if False:
        i = 10
        return i + 15
    if isinstance(event, (AssetMaterialization, AssetObservation, ExpectationResult)):
        return event
    elif isinstance(event, MaterializeResult):
        output_name = _output_name_for_result_obj(event, context)
        outputs_seen.add(output_name)
        return event
    else:
        if not isinstance(event, (Output, DynamicOutput)):
            raise DagsterInvariantViolationError(f'When yielding outputs from a {op_def.node_type_str} generator, they should be wrapped in an `Output` object.')
        else:
            output_def = output_defs[event.output_name]
            _type_check_output(output_def, event, context)
            if output_def.name in outputs_seen and (not isinstance(output_def, DynamicOutputDefinition)):
                raise DagsterInvariantViolationError(f"Invocation of {op_def.node_type_str} '{context.alias}' yielded an output '{output_def.name}' multiple times.")
            outputs_seen.add(output_def.name)
        return event

def _type_check_output_wrapper(op_def: 'OpDefinition', result: Any, context: 'BoundOpExecutionContext') -> Any:
    if False:
        print('Hello World!')
    'Type checks and returns the result of a op.\n\n    If the op result is itself a generator, then wrap in a fxn that will type check and yield\n    outputs.\n    '
    output_defs = {output_def.name: output_def for output_def in op_def.output_defs}
    if inspect.isasyncgen(result):

        async def to_gen(async_gen):
            outputs_seen = set()
            async for event in async_gen:
                yield _handle_gen_event(event, op_def, context, output_defs, outputs_seen)
            for output_def in op_def.output_defs:
                if output_def.name not in outputs_seen and output_def.is_required and (not output_def.is_dynamic):
                    if output_def.dagster_type.is_nothing:
                        yield Output(output_name=output_def.name, value=None)
                    else:
                        raise DagsterInvariantViolationError(f"Invocation of {op_def.node_type_str} '{context.alias}' did not return an output for non-optional output '{output_def.name}'")
        return to_gen(result)
    elif inspect.iscoroutine(result):

        async def type_check_coroutine(coro):
            out = await coro
            return _type_check_function_output(op_def, out, context)
        return type_check_coroutine(result)
    elif inspect.isgenerator(result):

        def type_check_gen(gen):
            if False:
                for i in range(10):
                    print('nop')
            outputs_seen = set()
            for event in gen:
                yield _handle_gen_event(event, op_def, context, output_defs, outputs_seen)
            for output_def in op_def.output_defs:
                if output_def.name not in outputs_seen and output_def.is_required and (not output_def.is_dynamic):
                    if output_def.dagster_type.is_nothing:
                        yield Output(output_name=output_def.name, value=None)
                    else:
                        raise DagsterInvariantViolationError(f'Invocation of {op_def.node_type_str} "{context.alias}" did not return an output for non-optional output "{output_def.name}"')
        return type_check_gen(result)
    return _type_check_function_output(op_def, result, context)

def _type_check_function_output(op_def: 'OpDefinition', result: T, context: 'BoundOpExecutionContext') -> T:
    if False:
        for i in range(10):
            print('nop')
    from ..execution.plan.compute_generator import validate_and_coerce_op_result_to_iterator
    output_defs_by_name = {output_def.name: output_def for output_def in op_def.output_defs}
    for event in validate_and_coerce_op_result_to_iterator(result, context, op_def.output_defs):
        if isinstance(event, (Output, DynamicOutput)):
            _type_check_output(output_defs_by_name[event.output_name], event, context)
        elif isinstance(event, MaterializeResult):
            _output_name_for_result_obj(event, context)
    return result

def _type_check_output(output_def: 'OutputDefinition', output: Union[Output, DynamicOutput], context: 'BoundOpExecutionContext') -> None:
    if False:
        print('Hello World!')
    'Validates and performs core type check on a provided output.\n\n    Args:\n        output_def (OutputDefinition): The output definition to validate against.\n        output (Any): The output to validate.\n        context (BoundOpExecutionContext): Context containing resources to be used for type\n            check.\n    '
    from ..execution.plan.execute_step import do_type_check
    op_label = context.describe_op()
    dagster_type = output_def.dagster_type
    type_check = do_type_check(context.for_type(dagster_type), dagster_type, output.value)
    if not type_check.success:
        raise DagsterTypeCheckDidNotPass(description=f'Type check failed for {op_label} output "{output.output_name}" - expected type "{dagster_type.display_name}". Description: {type_check.description}', metadata=type_check.metadata, dagster_type=dagster_type)
    context.observe_output(output_def.name, output.mapping_key if isinstance(output, DynamicOutput) else None)