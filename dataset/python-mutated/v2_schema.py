import inspect
import typing
import typing as t
import pendulum
import pydantic
from pydantic import BaseModel as V2BaseModel
from pydantic import ConfigDict, create_model
from pydantic.type_adapter import TypeAdapter
from prefect._internal.pydantic.annotations.pendulum import PydanticPendulumDateTimeType, PydanticPendulumDateType, PydanticPendulumDurationType
from prefect._internal.pydantic.schemas import GenerateEmptySchemaForUserClasses

def _is_v2_model(v) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(v, V2BaseModel):
        return True
    try:
        if inspect.isclass(v) and issubclass(v, V2BaseModel):
            return True
    except TypeError:
        pass

def has_v2_model_as_param(signature: inspect.Signature) -> bool:
    if False:
        return 10
    parameters = signature.parameters.values()
    for p in parameters:
        if _is_v2_model(p.annotation):
            return True
        for v in typing.get_args(p.annotation):
            if _is_v2_model(v):
                return True
    return False

def process_v2_params(param: inspect.Parameter, *, position: int, docstrings: t.Dict[str, str], aliases: t.Dict) -> t.Tuple[str, t.Any, 'pydantic.Field']:
    if False:
        while True:
            i = 10
    '\n    Generate a sanitized name, type, and pydantic.Field for a given parameter.\n\n    This implementation is exactly the same as the v1 implementation except\n    that it uses pydantic v2 constructs.\n    '
    if hasattr(pydantic.BaseModel, param.name):
        name = param.name + '__'
        aliases[name] = param.name
    else:
        name = param.name
    type_ = t.Any if param.annotation is inspect._empty else param.annotation
    if type_ == pendulum.DateTime:
        type_ = PydanticPendulumDateTimeType
    if type_ == pendulum.Date:
        type_ = PydanticPendulumDateType
    if type_ == pendulum.Duration:
        type_ = PydanticPendulumDurationType
    field = pydantic.Field(default=... if param.default is param.empty else param.default, title=param.name, description=docstrings.get(param.name, None), alias=aliases.get(name), json_schema_extra={'position': position})
    return (name, type_, field)

def create_v2_schema(name_: str, model_cfg: ConfigDict, **model_fields):
    if False:
        print('Hello World!')
    '\n    Create a pydantic v2 model and craft a v1 compatible schema from it.\n    '
    model = create_model(name_, __config__=model_cfg, **model_fields)
    adapter = TypeAdapter(model)
    schema = adapter.json_schema(by_alias=True, ref_template='#/definitions/{model}', schema_generator=GenerateEmptySchemaForUserClasses)
    if '$defs' in schema:
        schema['definitions'] = schema['$defs']
    return schema