from __future__ import annotations
from collections import defaultdict
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable
from pydantic_core import CoreSchema, PydanticCustomError, to_jsonable_python
from pydantic_core import core_schema as cs
from ._fields import PydanticMetadata
if TYPE_CHECKING:
    from ..annotated_handlers import GetJsonSchemaHandler
STRICT = {'strict'}
SEQUENCE_CONSTRAINTS = {'min_length', 'max_length'}
INEQUALITY = {'le', 'ge', 'lt', 'gt'}
NUMERIC_CONSTRAINTS = {'multiple_of', 'allow_inf_nan', *INEQUALITY}
STR_CONSTRAINTS = {*SEQUENCE_CONSTRAINTS, *STRICT, 'strip_whitespace', 'to_lower', 'to_upper', 'pattern'}
BYTES_CONSTRAINTS = {*SEQUENCE_CONSTRAINTS, *STRICT}
LIST_CONSTRAINTS = {*SEQUENCE_CONSTRAINTS, *STRICT}
TUPLE_CONSTRAINTS = {*SEQUENCE_CONSTRAINTS, *STRICT}
SET_CONSTRAINTS = {*SEQUENCE_CONSTRAINTS, *STRICT}
DICT_CONSTRAINTS = {*SEQUENCE_CONSTRAINTS, *STRICT}
GENERATOR_CONSTRAINTS = {*SEQUENCE_CONSTRAINTS, *STRICT}
FLOAT_CONSTRAINTS = {*NUMERIC_CONSTRAINTS, *STRICT}
INT_CONSTRAINTS = {*NUMERIC_CONSTRAINTS, *STRICT}
BOOL_CONSTRAINTS = STRICT
UUID_CONSTRAINTS = STRICT
DATE_TIME_CONSTRAINTS = {*NUMERIC_CONSTRAINTS, *STRICT}
TIMEDELTA_CONSTRAINTS = {*NUMERIC_CONSTRAINTS, *STRICT}
TIME_CONSTRAINTS = {*NUMERIC_CONSTRAINTS, *STRICT}
LAX_OR_STRICT_CONSTRAINTS = STRICT
UNION_CONSTRAINTS = {'union_mode'}
URL_CONSTRAINTS = {'max_length', 'allowed_schemes', 'host_required', 'default_host', 'default_port', 'default_path'}
TEXT_SCHEMA_TYPES = ('str', 'bytes', 'url', 'multi-host-url')
SEQUENCE_SCHEMA_TYPES = ('list', 'tuple', 'set', 'frozenset', 'generator', *TEXT_SCHEMA_TYPES)
NUMERIC_SCHEMA_TYPES = ('float', 'int', 'date', 'time', 'timedelta', 'datetime')
CONSTRAINTS_TO_ALLOWED_SCHEMAS: dict[str, set[str]] = defaultdict(set)
for constraint in STR_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(TEXT_SCHEMA_TYPES)
for constraint in BYTES_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('bytes',))
for constraint in LIST_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('list',))
for constraint in TUPLE_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('tuple',))
for constraint in SET_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('set', 'frozenset'))
for constraint in DICT_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('dict',))
for constraint in GENERATOR_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('generator',))
for constraint in FLOAT_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('float',))
for constraint in INT_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('int',))
for constraint in DATE_TIME_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('date', 'time', 'datetime'))
for constraint in TIMEDELTA_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('timedelta',))
for constraint in TIME_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('time',))
for schema_type in (*TEXT_SCHEMA_TYPES, *SEQUENCE_SCHEMA_TYPES, *NUMERIC_SCHEMA_TYPES, 'typed-dict', 'model'):
    CONSTRAINTS_TO_ALLOWED_SCHEMAS['strict'].add(schema_type)
for constraint in UNION_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('union',))
for constraint in URL_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('url', 'multi-host-url'))
for constraint in BOOL_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('bool',))
for constraint in UUID_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('uuid',))
for constraint in LAX_OR_STRICT_CONSTRAINTS:
    CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint].update(('lax-or-strict',))

def add_js_update_schema(s: cs.CoreSchema, f: Callable[[], dict[str, Any]]) -> None:
    if False:
        for i in range(10):
            print('nop')

    def update_js_schema(s: cs.CoreSchema, handler: GetJsonSchemaHandler) -> dict[str, Any]:
        if False:
            return 10
        js_schema = handler(s)
        js_schema.update(f())
        return js_schema
    if 'metadata' in s:
        metadata = s['metadata']
        if 'pydantic_js_functions' in s:
            metadata['pydantic_js_functions'].append(update_js_schema)
        else:
            metadata['pydantic_js_functions'] = [update_js_schema]
    else:
        s['metadata'] = {'pydantic_js_functions': [update_js_schema]}

def as_jsonable_value(v: Any) -> Any:
    if False:
        i = 10
        return i + 15
    if type(v) not in (int, str, float, bytes, bool, type(None)):
        return to_jsonable_python(v)
    return v

def expand_grouped_metadata(annotations: Iterable[Any]) -> Iterable[Any]:
    if False:
        i = 10
        return i + 15
    'Expand the annotations.\n\n    Args:\n        annotations: An iterable of annotations.\n\n    Returns:\n        An iterable of expanded annotations.\n\n    Example:\n        ```py\n        from annotated_types import Ge, Len\n\n        from pydantic._internal._known_annotated_metadata import expand_grouped_metadata\n\n        print(list(expand_grouped_metadata([Ge(4), Len(5)])))\n        #> [Ge(ge=4), MinLen(min_length=5)]\n        ```\n    '
    import annotated_types as at
    from pydantic.fields import FieldInfo
    for annotation in annotations:
        if isinstance(annotation, at.GroupedMetadata):
            yield from annotation
        elif isinstance(annotation, FieldInfo):
            yield from annotation.metadata
            annotation = copy(annotation)
            annotation.metadata = []
            yield annotation
        else:
            yield annotation

def apply_known_metadata(annotation: Any, schema: CoreSchema) -> CoreSchema | None:
    if False:
        return 10
    'Apply `annotation` to `schema` if it is an annotation we know about (Gt, Le, etc.).\n    Otherwise return `None`.\n\n    This does not handle all known annotations. If / when it does, it can always\n    return a CoreSchema and return the unmodified schema if the annotation should be ignored.\n\n    Assumes that GroupedMetadata has already been expanded via `expand_grouped_metadata`.\n\n    Args:\n        annotation: The annotation.\n        schema: The schema.\n\n    Returns:\n        An updated schema with annotation if it is an annotation we know about, `None` otherwise.\n\n    Raises:\n        PydanticCustomError: If `Predicate` fails.\n    '
    import annotated_types as at
    from . import _validators
    schema = schema.copy()
    (schema_update, other_metadata) = collect_known_metadata([annotation])
    schema_type = schema['type']
    for (constraint, value) in schema_update.items():
        if constraint not in CONSTRAINTS_TO_ALLOWED_SCHEMAS:
            raise ValueError(f'Unknown constraint {constraint}')
        allowed_schemas = CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint]
        if schema_type in allowed_schemas:
            if constraint == 'union_mode' and schema_type == 'union':
                schema['mode'] = value
            else:
                schema[constraint] = value
            continue
        if constraint == 'allow_inf_nan' and value is False:
            return cs.no_info_after_validator_function(_validators.forbid_inf_nan_check, schema)
        elif constraint == 'pattern':
            return cs.chain_schema([schema, cs.str_schema(pattern=value)])
        elif constraint == 'gt':
            s = cs.no_info_after_validator_function(partial(_validators.greater_than_validator, gt=value), schema)
            add_js_update_schema(s, lambda : {'gt': as_jsonable_value(value)})
            return s
        elif constraint == 'ge':
            return cs.no_info_after_validator_function(partial(_validators.greater_than_or_equal_validator, ge=value), schema)
        elif constraint == 'lt':
            return cs.no_info_after_validator_function(partial(_validators.less_than_validator, lt=value), schema)
        elif constraint == 'le':
            return cs.no_info_after_validator_function(partial(_validators.less_than_or_equal_validator, le=value), schema)
        elif constraint == 'multiple_of':
            return cs.no_info_after_validator_function(partial(_validators.multiple_of_validator, multiple_of=value), schema)
        elif constraint == 'min_length':
            s = cs.no_info_after_validator_function(partial(_validators.min_length_validator, min_length=value), schema)
            add_js_update_schema(s, lambda : {'minLength': as_jsonable_value(value)})
            return s
        elif constraint == 'max_length':
            s = cs.no_info_after_validator_function(partial(_validators.max_length_validator, max_length=value), schema)
            add_js_update_schema(s, lambda : {'maxLength': as_jsonable_value(value)})
            return s
        elif constraint == 'strip_whitespace':
            return cs.chain_schema([schema, cs.str_schema(strip_whitespace=True)])
        elif constraint == 'to_lower':
            return cs.chain_schema([schema, cs.str_schema(to_lower=True)])
        elif constraint == 'to_upper':
            return cs.chain_schema([schema, cs.str_schema(to_upper=True)])
        elif constraint == 'min_length':
            return cs.no_info_after_validator_function(partial(_validators.min_length_validator, min_length=annotation.min_length), schema)
        elif constraint == 'max_length':
            return cs.no_info_after_validator_function(partial(_validators.max_length_validator, max_length=annotation.max_length), schema)
        else:
            raise RuntimeError(f'Unable to apply constraint {constraint} to schema {schema_type}')
    for annotation in other_metadata:
        if isinstance(annotation, at.Gt):
            return cs.no_info_after_validator_function(partial(_validators.greater_than_validator, gt=annotation.gt), schema)
        elif isinstance(annotation, at.Ge):
            return cs.no_info_after_validator_function(partial(_validators.greater_than_or_equal_validator, ge=annotation.ge), schema)
        elif isinstance(annotation, at.Lt):
            return cs.no_info_after_validator_function(partial(_validators.less_than_validator, lt=annotation.lt), schema)
        elif isinstance(annotation, at.Le):
            return cs.no_info_after_validator_function(partial(_validators.less_than_or_equal_validator, le=annotation.le), schema)
        elif isinstance(annotation, at.MultipleOf):
            return cs.no_info_after_validator_function(partial(_validators.multiple_of_validator, multiple_of=annotation.multiple_of), schema)
        elif isinstance(annotation, at.MinLen):
            return cs.no_info_after_validator_function(partial(_validators.min_length_validator, min_length=annotation.min_length), schema)
        elif isinstance(annotation, at.MaxLen):
            return cs.no_info_after_validator_function(partial(_validators.max_length_validator, max_length=annotation.max_length), schema)
        elif isinstance(annotation, at.Predicate):
            predicate_name = f'{annotation.func.__qualname__} ' if hasattr(annotation.func, '__qualname__') else ''

            def val_func(v: Any) -> Any:
                if False:
                    return 10
                if not annotation.func(v):
                    raise PydanticCustomError('predicate_failed', f'Predicate {predicate_name}failed')
                return v
            return cs.no_info_after_validator_function(val_func, schema)
        return None
    return schema

def collect_known_metadata(annotations: Iterable[Any]) -> tuple[dict[str, Any], list[Any]]:
    if False:
        for i in range(10):
            print('nop')
    "Split `annotations` into known metadata and unknown annotations.\n\n    Args:\n        annotations: An iterable of annotations.\n\n    Returns:\n        A tuple contains a dict of known metadata and a list of unknown annotations.\n\n    Example:\n        ```py\n        from annotated_types import Gt, Len\n\n        from pydantic._internal._known_annotated_metadata import collect_known_metadata\n\n        print(collect_known_metadata([Gt(1), Len(42), ...]))\n        #> ({'gt': 1, 'min_length': 42}, [Ellipsis])\n        ```\n    "
    import annotated_types as at
    annotations = expand_grouped_metadata(annotations)
    res: dict[str, Any] = {}
    remaining: list[Any] = []
    for annotation in annotations:
        if isinstance(annotation, PydanticMetadata):
            res.update(annotation.__dict__)
        elif isinstance(annotation, at.MinLen):
            res.update({'min_length': annotation.min_length})
        elif isinstance(annotation, at.MaxLen):
            res.update({'max_length': annotation.max_length})
        elif isinstance(annotation, at.Gt):
            res.update({'gt': annotation.gt})
        elif isinstance(annotation, at.Ge):
            res.update({'ge': annotation.ge})
        elif isinstance(annotation, at.Lt):
            res.update({'lt': annotation.lt})
        elif isinstance(annotation, at.Le):
            res.update({'le': annotation.le})
        elif isinstance(annotation, at.MultipleOf):
            res.update({'multiple_of': annotation.multiple_of})
        elif isinstance(annotation, type) and issubclass(annotation, PydanticMetadata):
            res.update({k: v for (k, v) in vars(annotation).items() if not k.startswith('_')})
        else:
            remaining.append(annotation)
    res = {k: v for (k, v) in res.items() if v is not None}
    return (res, remaining)

def check_metadata(metadata: dict[str, Any], allowed: Iterable[str], source_type: Any) -> None:
    if False:
        i = 10
        return i + 15
    "A small utility function to validate that the given metadata can be applied to the target.\n    More than saving lines of code, this gives us a consistent error message for all of our internal implementations.\n\n    Args:\n        metadata: A dict of metadata.\n        allowed: An iterable of allowed metadata.\n        source_type: The source type.\n\n    Raises:\n        TypeError: If there is metadatas that can't be applied on source type.\n    "
    unknown = metadata.keys() - set(allowed)
    if unknown:
        raise TypeError(f"The following constraints cannot be applied to {source_type!r}: {', '.join([f'{k!r}' for k in unknown])}")