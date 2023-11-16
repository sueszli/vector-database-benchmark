import inspect
import json
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from typing import Callable, Generic, List, Optional, Sequence, Tuple, Type, TypeVar, Union
from django.http import HttpRequest
from django.utils.translation import gettext as _
from pydantic import Json, StringConstraints, TypeAdapter, ValidationError
from typing_extensions import Annotated, Concatenate, ParamSpec, TypeAlias, get_args, get_origin, get_type_hints
from zerver.lib.exceptions import ApiParamValidationError, JsonableError
from zerver.lib.request import _REQ, RequestConfusingParamsError, RequestNotes, RequestVariableMissingError, arguments_map
from zerver.lib.response import MutableJsonResponse
T = TypeVar('T')
ParamT = ParamSpec('ParamT')
ReturnT = TypeVar('ReturnT')

class DocumentationStatus(Enum):
    DOCUMENTED = auto()
    INTENTIONALLY_UNDOCUMENTED = auto()
    DOCUMENTATION_PENDING = auto()
DOCUMENTED = DocumentationStatus.DOCUMENTED
INTENTIONALLY_UNDOCUMENTED = DocumentationStatus.INTENTIONALLY_UNDOCUMENTED
DOCUMENTATION_PENDING = DocumentationStatus.DOCUMENTATION_PENDING

@dataclass(frozen=True)
class ApiParamConfig:
    """The metadata associated with a view function parameter as an annotation
    to configure how the typed_endpoint decorator should process it.

    It should be used with Annotated as the type annotation of a parameter
    in a @typed_endpoint-decorated function:
    ```
    @typed_endpoint
    def view(
        request: HttpRequest,
        *,
        flag_value: Annotated[Json[bool], ApiParamConfig(
            whence="flag",
            documentation_status=INTENTIONALLY_UNDOCUMENTED,
        )]
    ) -> HttpResponse:
        ...
    ```

    For a parameter that is not annotated with ApiParamConfig, typed_endpoint
    will construct a configuration using the defaults.

    whence:
    The name of the request variable that should be used for this parameter.
    If None, it is set to the name of the function parameter.

    path_only:
    Used for parameters included in the URL.

    argument_type_is_body:
    When set to true, the value of the parameter will be extracted from the
    request body instead of a single query parameter.

    documentation_status:
    The OpenAPI documentation status of this parameter. Unless it is set to
    INTENTIONALLY_UNDOCUMENTED or DOCUMENTATION_PENDING, the test suite is
    configured to raise an error when its documentation cannot be found.

    aliases:
    The names allowed for the request variable other than that specified with
    "whence".
    """
    whence: Optional[str] = None
    path_only: bool = False
    argument_type_is_body: bool = False
    documentation_status: DocumentationStatus = DOCUMENTED
    aliases: Tuple[str, ...] = ()
JsonBodyPayload: TypeAlias = Annotated[Json[T], ApiParamConfig(argument_type_is_body=True)]
PathOnly: TypeAlias = Annotated[T, ApiParamConfig(path_only=True)]
OptionalTopic: TypeAlias = Annotated[Optional[str], StringConstraints(strip_whitespace=True), ApiParamConfig(whence='topic', aliases=('subject',))]
RequiredStringConstraint = lambda : StringConstraints(strip_whitespace=True, min_length=1)

class _NotSpecified:
    pass
NotSpecified = _NotSpecified()

@dataclass(frozen=True)
class FuncParam(Generic[T]):
    default: Union[T, _NotSpecified]
    param_name: str
    param_type: Type[T]
    type_adapter: TypeAdapter[T]
    aliases: Tuple[str, ...]
    argument_type_is_body: bool
    documentation_status: DocumentationStatus
    path_only: bool
    request_var_name: str

@dataclass(frozen=True)
class ViewFuncInfo:
    view_func_full_name: str
    parameters: Sequence[FuncParam[object]]

def is_annotated(type_annotation: Type[object]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    origin = get_origin(type_annotation)
    return origin is Annotated

def is_optional(type_annotation: Type[object]) -> bool:
    if False:
        return 10
    origin = get_origin(type_annotation)
    type_args = get_args(type_annotation)
    return origin is Union and type(None) in type_args and (len(type_args) == 2)
API_PARAM_CONFIG_USAGE_HINT = f'\n    Detected incorrect usage of Annotated types for parameter {{param_name}}!\n    Check the placement of the {ApiParamConfig.__name__} object in the type annotation:\n\n    {{param_name}}: {{param_type}}\n\n    The Annotated[T, ...] type annotation containing the\n    {ApiParamConfig.__name__} object should not be nested inside another type.\n\n    Correct examples:\n\n    # Using Optional inside Annotated\n    param: Annotated[Optional[int], ApiParamConfig(...)]\n    param: Annotated[Optional[int], ApiParamConfig(...)]] = None\n\n    # Not using Optional when the default is not None\n    param: Annotated[int, ApiParamConfig(...)]\n\n    Incorrect examples:\n\n    # Nesting Annotated inside Optional\n    param: Optional[Annotated[int, ApiParamConfig(...)]]\n    param: Optional[Annotated[int, ApiParamConfig(...)]] = None\n\n    # Nesting the Annotated type carrying ApiParamConfig inside other types like Union\n    param: Union[str, Annotated[int, ApiParamConfig(...)]]\n'

def parse_single_parameter(param_name: str, param_type: Type[T], parameter: inspect.Parameter) -> FuncParam[T]:
    if False:
        for i in range(10):
            print('nop')
    param_default = parameter.default
    if param_default is inspect._empty:
        param_default = NotSpecified
    if param_default is None and is_optional(param_type):
        type_args = get_args(param_type)
        inner_type = type_args[0] if type_args[1] is type(None) else type_args[1]
        if is_annotated(inner_type):
            (annotated_type, *annotations) = get_args(inner_type)
            has_api_param_config = any((isinstance(annotation, ApiParamConfig) for annotation in annotations))
            assert not has_api_param_config or is_optional(annotated_type), API_PARAM_CONFIG_USAGE_HINT.format(param_name=param_name, param_type=param_type)
            param_type = inner_type
    param_config: Optional[ApiParamConfig] = None
    if is_annotated(param_type):
        (ignored_type, *annotations) = get_args(param_type)
        for annotation in annotations:
            if not isinstance(annotation, ApiParamConfig):
                continue
            assert param_config is None, 'ApiParamConfig can only be defined once per parameter'
            param_config = annotation
    else:
        assert ApiParamConfig.__name__ not in str(param_type), API_PARAM_CONFIG_USAGE_HINT.format(param_name=param_name, param_type=param_type)
    if param_config is None:
        param_config = ApiParamConfig()
    if param_config.argument_type_is_body:
        request_var_name = 'request'
    else:
        request_var_name = param_config.whence if param_config.whence is not None else param_name
    return FuncParam(default=param_default, param_name=param_name, param_type=param_type, type_adapter=TypeAdapter(param_type), aliases=param_config.aliases, argument_type_is_body=param_config.argument_type_is_body, documentation_status=param_config.documentation_status, path_only=param_config.path_only, request_var_name=request_var_name)

def parse_view_func_signature(view_func: Callable[Concatenate[HttpRequest, ParamT], object]) -> ViewFuncInfo:
    if False:
        while True:
            i = 10
    'This is responsible for inspecting the function signature and getting the\n    metadata from the parameters. We want to keep this function as pure as\n    possible not leaking side effects to the global state. Side effects should\n    be executed separately after the ViewFuncInfo is returned.\n    '
    type_hints = get_type_hints(view_func, include_extras=True)
    parameters = inspect.signature(view_func).parameters
    view_func_full_name = f'{view_func.__module__}.{view_func.__name__}'
    process_parameters: List[FuncParam[object]] = []
    for (param_name, parameter) in parameters.items():
        assert param_name in type_hints
        if parameter.kind != inspect.Parameter.KEYWORD_ONLY:
            continue
        param_info = parse_single_parameter(param_name=param_name, param_type=type_hints[param_name], parameter=parameter)
        process_parameters.append(param_info)
    return ViewFuncInfo(view_func_full_name=view_func_full_name, parameters=process_parameters)
ERROR_TEMPLATES = {'bool_parsing': _('{var_name} is not a boolean'), 'bool_type': _('{var_name} is not a boolean'), 'datetime_parsing': _('{var_name} is not a date'), 'datetime_type': _('{var_name} is not a date'), 'dict_type': _('{var_name} is not a dict'), 'extra_forbidden': _('Argument "{argument}" at {var_name} is unexpected'), 'float_parsing': _('{var_name} is not a float'), 'float_type': _('{var_name} is not a float'), 'greater_than': _('{var_name} is too small'), 'int_parsing': _('{var_name} is not an integer'), 'int_type': _('{var_name} is not an integer'), 'json_invalid': _('{var_name} is not valid JSON'), 'json_type': _('{var_name} is not valid JSON'), 'less_than': _('{var_name} is too large'), 'list_type': _('{var_name} is not a list'), 'literal_error': _('Invalid {var_name}'), 'string_too_long': _('{var_name} is too long (limit: {max_length} characters)'), 'string_too_short': _('{var_name} is too short.'), 'string_type': _('{var_name} is not a string'), 'unexpected_keyword_argument': _('Argument "{argument}" at {var_name} is unexpected')}

def parse_value_for_parameter(parameter: FuncParam[T], value: object) -> T:
    if False:
        return 10
    try:
        return parameter.type_adapter.validate_python(value, strict=True)
    except ValidationError as exc:
        error = exc.errors()[0]
        error_template = ERROR_TEMPLATES.get(error['type'])
        var_name = parameter.request_var_name + ''.join((f'[{json.dumps(loc)}]' for loc in error['loc']))
        context = {'var_name': var_name, **error.get('ctx', {})}
        if error['type'] == 'json_invalid' and parameter.argument_type_is_body:
            error_template = _('Malformed JSON')
        elif error['type'] in ('unexpected_keyword_argument', 'extra_forbidden'):
            context['argument'] = error['loc'][-1]
        elif error['type'] == 'string_too_short' and error['ctx'].get('min_length') == 1:
            error_template = _('{var_name} cannot be blank')
        assert error_template is not None, MISSING_ERROR_TEMPLATE.format(error_type=error['type'], url=error.get('url', '(documentation unavailable)'), error=json.dumps(error, indent=4))
        raise ApiParamValidationError(error_template.format(**context), error['type'])
MISSING_ERROR_TEMPLATE = f'\n    Pydantic validation error of type "{{error_type}}" does not have the\n    corresponding error message template or is not handled explicitly. We expect\n    that every validation error is formatted into a client-facing error message.\n    Consider adding this type to {__package__}.ERROR_TEMPLATES with the appropriate\n    internationalized error message or handle it in {__package__}.{parse_value_for_parameter.__name__}.\n\n    Documentation for "{{error_type}}" can be found at {{url}}.\n\n    Error information:\n{{error}}\n'
UNEXPECTEDLY_MISSING_KEYWORD_ONLY_PARAMETERS = '\nParameters expected to be parsed from the request should be defined as\nkeyword-only parameters, but there is no keyword-only parameter found in\n{view_func_name}.\n\nExample usage:\n\n```\n@typed_endpoint\ndef view(\n    request: HttpRequest,\n    *,\n    flag_value: Annotated[Json[bool], ApiParamConfig(\n        whence="flag", documentation_status=INTENTIONALLY_UNDOCUMENTED,\n    )]\n) -> HttpResponse:\n    ...\n```\n\nThis is likely a programming error. See https://peps.python.org/pep-3102/ for details on how\nto correctly declare your parameters as keyword-only parameters.\nEndpoints that do not accept parameters should use @typed_endpoint_without_parameters.\n'
UNEXPECTED_KEYWORD_ONLY_PARAMETERS = '\nUnexpected keyword-only parameters found in {view_func_name}.\nkeyword-only parameters are treated as parameters to be parsed from the request,\nbut @typed_endpoint_without_parameters does not expect any.\n\nUse @typed_endpoint instead.\n'

def typed_endpoint_without_parameters(view_func: Callable[Concatenate[HttpRequest, ParamT], ReturnT]) -> Callable[Concatenate[HttpRequest, ParamT], ReturnT]:
    if False:
        for i in range(10):
            print('nop')
    return typed_endpoint(view_func, expect_no_parameters=True)

def typed_endpoint(view_func: Callable[Concatenate[HttpRequest, ParamT], ReturnT], *, expect_no_parameters: bool=False) -> Callable[Concatenate[HttpRequest, ParamT], ReturnT]:
    if False:
        while True:
            i = 10
    endpoint_info = parse_view_func_signature(view_func)
    if expect_no_parameters:
        assert len(endpoint_info.parameters) == 0, UNEXPECTED_KEYWORD_ONLY_PARAMETERS.format(view_func_name=endpoint_info.view_func_full_name)
    else:
        assert len(endpoint_info.parameters) != 0, UNEXPECTEDLY_MISSING_KEYWORD_ONLY_PARAMETERS.format(view_func_name=endpoint_info.view_func_full_name)
    for func_param in endpoint_info.parameters:
        assert not isinstance(func_param.default, _REQ), f'Unexpected REQ for parameter {func_param.param_name}; REQ is incompatible with typed_endpoint'
        if func_param.path_only:
            assert func_param.default is NotSpecified, f'Path-only parameter {func_param.param_name} should not have a default value'
        if func_param.documentation_status is DocumentationStatus.DOCUMENTED and (not func_param.path_only):
            arguments_map[endpoint_info.view_func_full_name].append(func_param.request_var_name)

    @wraps(view_func)
    def _wrapped_view_func(request: HttpRequest, /, *args: ParamT.args, **kwargs: ParamT.kwargs) -> ReturnT:
        if False:
            print('Hello World!')
        request_notes = RequestNotes.get_notes(request)
        for parameter in endpoint_info.parameters:
            if parameter.path_only:
                assert parameter.param_name in kwargs, f'Path-only variable {parameter.param_name} should be passed already'
            if parameter.param_name in kwargs:
                continue
            if parameter.argument_type_is_body:
                try:
                    request_notes.processed_parameters.add(parameter.request_var_name)
                    kwargs[parameter.param_name] = parse_value_for_parameter(parameter, request.body.decode(request.encoding or 'utf-8'))
                except UnicodeDecodeError:
                    raise JsonableError(_('Malformed payload'))
                continue
            possible_aliases = [parameter.request_var_name, *parameter.aliases]
            alias_used = None
            value_to_parse = None
            for current_alias in possible_aliases:
                if current_alias in request.POST:
                    value_to_parse = request.POST[current_alias]
                elif current_alias in request.GET:
                    value_to_parse = request.GET[current_alias]
                else:
                    continue
                if alias_used is not None:
                    raise RequestConfusingParamsError(alias_used, current_alias)
                alias_used = current_alias
            if alias_used is None:
                alias_used = parameter.request_var_name
                if parameter.default is NotSpecified:
                    raise RequestVariableMissingError(alias_used)
                continue
            assert value_to_parse is not None
            request_notes.processed_parameters.add(alias_used)
            kwargs[parameter.param_name] = parse_value_for_parameter(parameter, value_to_parse)
        return_value = view_func(request, *args, **kwargs)
        if isinstance(return_value, MutableJsonResponse) and (not request_notes.is_webhook_view) and (200 <= return_value.status_code < 300):
            ignored_parameters = {*request.POST, *request.GET}.difference(request_notes.processed_parameters)
            if ignored_parameters:
                return_value.get_data()['ignored_parameters_unsupported'] = sorted(ignored_parameters)
            else:
                return_value.get_data().pop('ignored_parameters_unsupported', None)
        return return_value
    _wrapped_view_func.use_endpoint = True
    return _wrapped_view_func