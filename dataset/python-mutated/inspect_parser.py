import ast
import inspect
import types
from typing import Any, Callable, Dict, get_args, get_origin, List, Mapping, Optional, Union
import astunparse
from typing_extensions import Annotated
from .parameter import Parameter

def extract_qualified_name(callable_object: Callable[..., object]) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    if inspect.ismethod(callable_object):
        return extract_qualified_name(callable_object.__func__)
    else:
        module_name = getattr(callable_object, '__module__', None)
        if module_name is None and (objclass := getattr(callable_object, '__objclass__', None)):
            module_name = getattr(objclass, '__module__', None)
        view_name = getattr(callable_object, '__qualname__', callable_object.__class__.__qualname__)
    if '<locals>' in view_name:
        return None
    return '.'.join(filter(None, [module_name, view_name]))

def extract_parameters(callable_object: Callable[..., object], strip_annotated: bool=True) -> List[Parameter]:
    if False:
        for i in range(10):
            print('nop')
    callable_parameters: Mapping[str, inspect.Parameter] = {}
    if isinstance(callable_object, types.FunctionType):
        callable_parameters = inspect.signature(callable_object).parameters
    elif isinstance(callable_object, types.MethodType):
        callable_parameters = inspect.signature(callable_object.__func__).parameters
    parameters: List[Parameter] = []
    for parameter in callable_parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            kind = Parameter.Kind.KWARG
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            kind = Parameter.Kind.VARARG
        else:
            kind = Parameter.Kind.ARG
        parameters.append(Parameter(_extract_parameter_name(parameter), _extract_parameter_annotation(parameter, strip_annotated), kind))
    return parameters

def ast_to_pretty_string(ast_expression: ast.expr) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    This function unparse an expression and modifies the result to make it compatible with the type annotation syntax.\n    For example astunparse.unparse will return `Tuple[(int, int)]\n` when parsing a Tuple annotation.\n    This function converts this in Tuple[int, int] which is the valid type syntax\n    '
    return astunparse.unparse(ast_expression).strip().replace('[(', '[').replace(')]', ']')

def strip_custom_annotations(annotation: str) -> str:
    if False:
        return 10
    if annotation.startswith('Annotated['):
        parsed_annotation = ast.parse(annotation).body[0]
        if isinstance(parsed_annotation, ast.Expr) and isinstance(parsed_annotation.value, ast.Subscript):
            if isinstance(parsed_annotation.value.slice, ast.Tuple):
                return ast_to_pretty_string(parsed_annotation.value.slice.elts[0])
            if isinstance(parsed_annotation.value.slice, ast.Index) and isinstance(parsed_annotation.value.slice.value, ast.Tuple):
                return ast_to_pretty_string(parsed_annotation.value.slice.value.elts[0])
    return annotation

def _get_annotations_as_types(function: Callable[..., object], strip_annotated: bool=False, strip_optional: bool=False) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    try:
        from inspect import get_annotations
        resolved_annotations = get_annotations(function, eval_str=True)
    except ImportError:
        resolved_annotations = function.__annotations__
    finally:
        for (parameter, annotation) in resolved_annotations.items():
            annotation = _strip_optional_annotation_from_type(annotation) if strip_optional else annotation
            resolved_annotations[parameter] = _strip_annotated_annotation_from_type(annotation) if strip_annotated else annotation
    return resolved_annotations

def _strip_optional_annotation_from_type(annotation: Any) -> Any:
    if False:
        print('Hello World!')
    if get_origin(annotation) is Union and len(get_args(annotation)) == 2 and (get_args(annotation)[1] == type(None)):
        return get_args(annotation)[0]
    return annotation

def _strip_annotated_annotation_from_type(annotation: Any) -> Any:
    if False:
        i = 10
        return i + 15
    if isinstance(annotation, type(Annotated[int, 'test'])):
        return get_args(annotation)[0]
    return annotation

def extract_parameters_with_types(function: Callable[..., object], strip_annotated: bool=False, strip_optional: bool=False) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    annotated_parameters = _get_annotations_as_types(function, strip_annotated, strip_optional)
    if 'kwargs' in annotated_parameters:
        annotated_parameters['**kwargs'] = annotated_parameters.pop('kwargs')
    if 'args' in annotated_parameters:
        annotated_parameters['*args'] = annotated_parameters.pop('args')
    annotated_parameters.pop('return', None)
    all_parameters = [parameter.name for parameter in extract_parameters(function)]
    for parameter in all_parameters:
        if parameter not in annotated_parameters:
            annotated_parameters[parameter] = None
    return annotated_parameters

def _extract_parameter_annotation(parameter: inspect.Parameter, strip_annotated: bool) -> Optional[str]:
    if False:
        while True:
            i = 10
    annotation = parameter.annotation
    if isinstance(annotation, str):
        if strip_annotated:
            return strip_custom_annotations(annotation)
        else:
            return annotation
    elif isinstance(annotation, type):
        return annotation.__name__
    else:
        return None

def _extract_parameter_name(parameter: inspect.Parameter) -> str:
    if False:
        return 10
    kind = parameter.kind
    if kind == inspect.Parameter.VAR_KEYWORD:
        return f'**{parameter.name}'
    elif kind == inspect.Parameter.VAR_POSITIONAL:
        return f'*{parameter.name}'
    else:
        return f'{parameter.name}'