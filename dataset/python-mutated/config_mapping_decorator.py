from typing import Any, Callable, Optional, Type, Union, cast, overload
import dagster._check as check
from dagster._config import UserConfigSchema
from dagster._core.decorator_utils import get_function_params
from ..config import ConfigMapping, ConfigMappingFn

class _ConfigMapping:

    def __init__(self, config_schema: Optional[UserConfigSchema]=None, receive_processed_config_values: Optional[bool]=None):
        if False:
            while True:
                i = 10
        self.config_schema = config_schema
        self.receive_processed_config_values = check.opt_bool_param(receive_processed_config_values, 'receive_processed_config_values')

    def __call__(self, fn: Callable[..., Any]) -> ConfigMapping:
        if False:
            print('Hello World!')
        check.callable_param(fn, 'fn')
        from dagster._config.pythonic_config import Config, infer_schema_from_config_annotation, safe_is_subclass
        from dagster._core.definitions.run_config import RunConfig
        config_fn_params = get_function_params(fn)
        check.invariant(len(config_fn_params) == 1, 'Config mapping should have exactly one parameter')
        param = config_fn_params[0]
        if safe_is_subclass(param.annotation, Config):
            check.invariant(self.config_schema is None, 'Cannot provide config_schema to config mapping with Config-annotated param')
            config_schema = infer_schema_from_config_annotation(param.annotation, param.default)
            config_cls = cast(Type[Config], param.annotation)
            param_name = param.name

            def wrapped_fn(config_as_dict) -> Any:
                if False:
                    for i in range(10):
                        print('nop')
                config_input = config_cls(**config_as_dict)
                output = fn(**{param_name: config_input})
                if isinstance(output, RunConfig):
                    return output.to_config_dict()
                else:
                    return output
            return ConfigMapping(config_fn=wrapped_fn, config_schema=config_schema, receive_processed_config_values=None)
        return ConfigMapping(config_fn=fn, config_schema=self.config_schema, receive_processed_config_values=self.receive_processed_config_values)

@overload
def config_mapping(config_fn: ConfigMappingFn) -> ConfigMapping:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def config_mapping(*, config_schema: UserConfigSchema=..., receive_processed_config_values: Optional[bool]=...) -> Callable[[ConfigMappingFn], ConfigMapping]:
    if False:
        i = 10
        return i + 15
    ...

def config_mapping(config_fn: Optional[Callable[..., Any]]=None, *, config_schema: Optional[UserConfigSchema]=None, receive_processed_config_values: Optional[bool]=None) -> Union[Callable[[ConfigMappingFn], ConfigMapping], ConfigMapping]:
    if False:
        print('Hello World!')
    'Create a config mapping with the specified parameters from the decorated function.\n\n    The config schema will be inferred from the type signature of the decorated function if not explicitly provided.\n\n    Args:\n        config_schema (ConfigSchema): The schema of the composite config.\n        receive_processed_config_values (Optional[bool]): If true, config values provided to the config_fn\n            will be converted to their dagster types before being passed in. For example, if this\n            value is true, enum config passed to config_fn will be actual enums, while if false,\n            then enum config passed to config_fn will be strings.\n\n\n    Examples:\n        .. code-block:: python\n\n            @op\n            def my_op(context):\n                return context.op_config["foo"]\n\n            @graph\n            def my_graph():\n                my_op()\n\n            @config_mapping\n            def my_config_mapping(val):\n                return {"ops": {"my_op": {"config": {"foo": val["foo"]}}}}\n\n            @config_mapping(config_schema={"foo": str})\n            def my_config_mapping(val):\n                return {"ops": {"my_op": {"config": {"foo": val["foo"]}}}}\n\n            result = my_graph.to_job(config=my_config_mapping).execute_in_process()\n\n    '
    if config_fn is not None:
        check.invariant(config_schema is None)
        check.invariant(receive_processed_config_values is None)
        return _ConfigMapping()(config_fn)
    check.invariant(config_fn is None)
    return _ConfigMapping(config_schema=config_schema, receive_processed_config_values=receive_processed_config_values)