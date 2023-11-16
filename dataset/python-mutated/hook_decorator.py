from functools import update_wrapper
from typing import TYPE_CHECKING, AbstractSet, Any, Callable, Optional, Sequence, Union, cast, overload
import dagster._check as check
from dagster._core.errors import DagsterInvalidDefinitionError
from ...decorator_utils import get_function_params, validate_expected_params
from ..events import HookExecutionResult
from ..hook_definition import HookDefinition
if TYPE_CHECKING:
    from dagster._core.events import DagsterEvent
    from dagster._core.execution.context.hook import HookContext

def _validate_hook_fn_params(fn, expected_positionals):
    if False:
        return 10
    params = get_function_params(fn)
    missing_positional = validate_expected_params(params, expected_positionals)
    if missing_positional:
        raise DagsterInvalidDefinitionError(f"'{fn.__name__}' decorated function does not have required positional parameter '{missing_positional}'. Hook functions should only have keyword arguments that match input names and a first positional parameter named 'context' and a second positional parameter named 'event_list'.")

class _Hook:

    def __init__(self, name: Optional[str]=None, required_resource_keys: Optional[AbstractSet[str]]=None, decorated_fn: Optional[Callable[..., Any]]=None):
        if False:
            print('Hello World!')
        self.name = check.opt_str_param(name, 'name')
        self.required_resource_keys = check.opt_set_param(required_resource_keys, 'required_resource_keys')
        self.decorated_fn = check.opt_callable_param(decorated_fn, 'decorated_fn')

    def __call__(self, fn) -> HookDefinition:
        if False:
            i = 10
            return i + 15
        check.callable_param(fn, 'fn')
        if not self.name:
            self.name = fn.__name__
        expected_positionals = ['context', 'event_list']
        _validate_hook_fn_params(fn, expected_positionals)
        hook_def = HookDefinition(name=self.name or '', hook_fn=fn, required_resource_keys=self.required_resource_keys, decorated_fn=self.decorated_fn or fn)
        update_wrapper(cast(Callable[..., Any], hook_def), fn)
        return hook_def

@overload
def event_list_hook(hook_fn: Callable) -> HookDefinition:
    if False:
        for i in range(10):
            print('nop')
    pass

@overload
def event_list_hook(*, name: Optional[str]=..., required_resource_keys: Optional[AbstractSet[str]]=..., decorated_fn: Optional[Callable[..., Any]]=...) -> _Hook:
    if False:
        for i in range(10):
            print('nop')
    pass

def event_list_hook(hook_fn: Optional[Callable]=None, *, name: Optional[str]=None, required_resource_keys: Optional[AbstractSet[str]]=None, decorated_fn: Optional[Callable[..., Any]]=None) -> Union[HookDefinition, _Hook]:
    if False:
        i = 10
        return i + 15
    "Create a generic hook with the specified parameters from the decorated function.\n\n    This decorator is currently used internally by Dagster machinery to support success_hook and\n    failure_hook.\n\n    The user-defined hook function requires two parameters:\n    - A `context` object is passed as the first parameter. The context is an instance of\n        :py:class:`context <HookContext>`, and provides access to system\n        information, such as loggers (context.log), resources (context.resources), the op\n        (context.op) and its execution step (context.step) which triggers this hook.\n    - An `event_list` object is passed as the second paramter. It provides the full event list of the\n        associated execution step.\n\n    Args:\n        name (Optional[str]): The name of this hook.\n        required_resource_keys (Optional[AbstractSet[str]]): Keys for the resources required by the\n            hook.\n\n    Examples:\n        .. code-block:: python\n\n            @event_list_hook(required_resource_keys={'slack'})\n            def slack_on_materializations(context, event_list):\n                for event in event_list:\n                    if event.event_type == DagsterEventType.ASSET_MATERIALIZATION:\n                        message = f'{context.op_name} has materialized an asset {event.asset_key}.'\n                        # send a slack message every time a materialization event occurs\n                        context.resources.slack.send_message(message)\n\n\n    "
    if hook_fn is not None:
        check.invariant(required_resource_keys is None)
        return _Hook()(hook_fn)
    return _Hook(name=name, required_resource_keys=required_resource_keys, decorated_fn=decorated_fn)
SuccessOrFailureHookFn = Callable[['HookContext'], Any]

@overload
def success_hook(hook_fn: SuccessOrFailureHookFn) -> HookDefinition:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def success_hook(*, name: Optional[str]=..., required_resource_keys: Optional[AbstractSet[str]]=...) -> Callable[[SuccessOrFailureHookFn], HookDefinition]:
    if False:
        print('Hello World!')
    ...

def success_hook(hook_fn: Optional[SuccessOrFailureHookFn]=None, *, name: Optional[str]=None, required_resource_keys: Optional[AbstractSet[str]]=None) -> Union[HookDefinition, Callable[[SuccessOrFailureHookFn], HookDefinition]]:
    if False:
        i = 10
        return i + 15
    "Create a hook on step success events with the specified parameters from the decorated function.\n\n    Args:\n        name (Optional[str]): The name of this hook.\n        required_resource_keys (Optional[AbstractSet[str]]): Keys for the resources required by the\n            hook.\n\n    Examples:\n        .. code-block:: python\n\n            @success_hook(required_resource_keys={'slack'})\n            def slack_message_on_success(context):\n                message = 'op {} succeeded'.format(context.op.name)\n                context.resources.slack.send_message(message)\n\n            @success_hook\n            def do_something_on_success(context):\n                do_something()\n\n\n    "

    def wrapper(fn: SuccessOrFailureHookFn) -> HookDefinition:
        if False:
            i = 10
            return i + 15
        check.callable_param(fn, 'fn')
        expected_positionals = ['context']
        _validate_hook_fn_params(fn, expected_positionals)
        if name is None or callable(name):
            _name = fn.__name__
        else:
            _name = name

        @event_list_hook(name=_name, required_resource_keys=required_resource_keys, decorated_fn=fn)
        def _success_hook(context: 'HookContext', event_list: Sequence['DagsterEvent']) -> HookExecutionResult:
            if False:
                for i in range(10):
                    print('nop')
            for event in event_list:
                if event.is_step_success:
                    fn(context)
                    return HookExecutionResult(hook_name=_name, is_skipped=False)
            return HookExecutionResult(hook_name=_name, is_skipped=True)
        return _success_hook
    if hook_fn is not None:
        check.invariant(required_resource_keys is None)
        return wrapper(hook_fn)
    return wrapper

@overload
def failure_hook(name: SuccessOrFailureHookFn) -> HookDefinition:
    if False:
        while True:
            i = 10
    ...

@overload
def failure_hook(name: Optional[str]=..., required_resource_keys: Optional[AbstractSet[str]]=...) -> Callable[[SuccessOrFailureHookFn], HookDefinition]:
    if False:
        i = 10
        return i + 15
    ...

def failure_hook(name: Optional[Union[SuccessOrFailureHookFn, str]]=None, required_resource_keys: Optional[AbstractSet[str]]=None) -> Union[HookDefinition, Callable[[SuccessOrFailureHookFn], HookDefinition]]:
    if False:
        while True:
            i = 10
    "Create a hook on step failure events with the specified parameters from the decorated function.\n\n    Args:\n        name (Optional[str]): The name of this hook.\n        required_resource_keys (Optional[AbstractSet[str]]): Keys for the resources required by the\n            hook.\n\n    Examples:\n        .. code-block:: python\n\n            @failure_hook(required_resource_keys={'slack'})\n            def slack_message_on_failure(context):\n                message = 'op {} failed'.format(context.op.name)\n                context.resources.slack.send_message(message)\n\n            @failure_hook\n            def do_something_on_failure(context):\n                do_something()\n\n\n    "

    def wrapper(fn: Callable[['HookContext'], Any]) -> HookDefinition:
        if False:
            i = 10
            return i + 15
        check.callable_param(fn, 'fn')
        expected_positionals = ['context']
        _validate_hook_fn_params(fn, expected_positionals)
        if name is None or callable(name):
            _name = fn.__name__
        else:
            _name = name

        @event_list_hook(name=_name, required_resource_keys=required_resource_keys, decorated_fn=fn)
        def _failure_hook(context: 'HookContext', event_list: Sequence['DagsterEvent']) -> HookExecutionResult:
            if False:
                for i in range(10):
                    print('nop')
            for event in event_list:
                if event.is_step_failure:
                    fn(context)
                    return HookExecutionResult(hook_name=_name, is_skipped=False)
            return HookExecutionResult(hook_name=_name, is_skipped=True)
        return _failure_hook
    if callable(name):
        check.invariant(required_resource_keys is None)
        return wrapper(name)
    return wrapper