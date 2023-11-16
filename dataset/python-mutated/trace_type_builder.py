"""Utitiles for Cache Key generation based on Function Trace Type."""
import collections.abc
from typing import Any, Dict, Hashable, Optional
import weakref
from tensorflow.core.function.trace_type import custom_nest_trace_type
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
from tensorflow.python.util import custom_nest_protocol

class InternalTracingContext(trace.TracingContext):
    """Container for variables and flags shared across TraceType generation."""

    def __init__(self, is_legacy_signature: bool=False):
        if False:
            i = 10
            return i + 15
        self._global_to_local_id = {}
        self._alias_id_to_placeholder = {}
        self._is_legacy_signature = is_legacy_signature

    def alias_global_id(self, global_id: Hashable) -> Hashable:
        if False:
            for i in range(10):
                print('nop')
        if global_id not in self._global_to_local_id:
            self._global_to_local_id[global_id] = len(self._global_to_local_id)
        return self._global_to_local_id[global_id]

    def add_placeholder(self, alias_id: Hashable, variable) -> None:
        if False:
            print('Hello World!')
        self._alias_id_to_placeholder[alias_id] = variable

    def get_placeholder_mapping(self) -> Dict[Hashable, Any]:
        if False:
            while True:
                i = 10
        return self._alias_id_to_placeholder

    @property
    def is_legacy_signature(self) -> bool:
        if False:
            print('Hello World!')
        'If the value is from a legacy signature representation.\n\n    Legacy signature representations include tf.function.input_signature and\n    ConcreteFunction.structured_input_signature.\n    '
        return self._is_legacy_signature

class InternalPlaceholderContext(trace.PlaceholderContext):
    """Container with mappings shared across TraceTypes for placeholder values."""

    def __init__(self, context_graph=None, placeholder_mapping=None, unnest_only=False, with_none_control_dependencies=False, composite_device_name=None):
        if False:
            i = 10
            return i + 15
        self._alias_id_to_placeholder = placeholder_mapping or {}
        self._naming_scope = None
        self._context_graph = context_graph
        self._unnest_only = unnest_only
        self._with_none_control_dependencies = with_none_control_dependencies
        self._composite_device_name = composite_device_name

    def has_placeholder(self, alias_id: Hashable) -> bool:
        if False:
            i = 10
            return i + 15
        return alias_id in self._alias_id_to_placeholder

    def get_placeholder(self, alias_id: Hashable) -> Hashable:
        if False:
            print('Hello World!')
        if not self.has_placeholder(alias_id):
            raise KeyError(f'alias_id: {alias_id} not found in this instance of placeholder context.')
        return self._alias_id_to_placeholder[alias_id]

    def add_placeholder(self, alias_id: Hashable, placeholder: Hashable) -> None:
        if False:
            while True:
                i = 10
        if alias_id in self._alias_id_to_placeholder:
            raise KeyError(f'alias id: {alias_id} is already stored in this instance of placeholder context.')
        self._alias_id_to_placeholder[alias_id] = placeholder

    def update_naming_scope(self, naming_scope: Optional[str]) -> None:
        if False:
            i = 10
            return i + 15
        self._naming_scope = naming_scope

    @property
    def naming_scope(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._naming_scope

    @property
    def context_graph(self):
        if False:
            i = 10
            return i + 15
        return self._context_graph

    @property
    def unnest_only(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._unnest_only

    @property
    def with_none_control_dependencies(self) -> bool:
        if False:
            print('Hello World!')
        return self._with_none_control_dependencies

    @property
    def composite_device_name(self) -> Any:
        if False:
            return 10
        return self._composite_device_name

class InternalCastContext(trace.CastContext):
    """Default casting behaviors."""

    def __init__(self, allow_specs=False):
        if False:
            while True:
                i = 10
        self._allow_specs = allow_specs

    @property
    def allow_specs(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Allow TypeSpecs to be casted (instead of the actual CompositeTensors).'
        return self._allow_specs

def from_value(value: Any, context: trace.TracingContext=None) -> trace.TraceType:
    if False:
        i = 10
        return i + 15
    'Returns a TraceType corresponding to the value based on the context.\n\n  Args:\n    value: The value to generate a TraceType for.\n    context: The TracingContext to be shared during protocol calls.\n\n  Returns:\n    A TraceType object representing the given value.\n  '
    if context is None:
        context = InternalTracingContext()
    if context.is_legacy_signature and isinstance(value, trace.TraceType):
        return value
    elif isinstance(value, trace.SupportsTracingProtocol):
        generated_type = value.__tf_tracing_type__(context)
        if not isinstance(generated_type, trace.TraceType):
            raise TypeError('Expected an instance of TraceType for Tracing Protocol call to ' + str(value) + ' but got ' + str(generated_type))
        return generated_type
    if isinstance(value, weakref.ref):
        raise TypeError(f'weakref input {value} not supported for tf.function.')
    if hasattr(value, '__wrapped__'):
        return from_value(value.__wrapped__, context)
    if isinstance(value, list):
        return default_types.List(*(from_value(c, context) for c in value))
    if isinstance(value, tuple):
        if util.is_namedtuple(value):
            named_tuple_type = type(value)
            return default_types.NamedTuple.from_type_and_attributes(named_tuple_type, tuple((from_value(c, context) for c in value)))
        else:
            return default_types.Tuple(*(from_value(c, context) for c in value))
    if isinstance(value, collections.abc.Mapping):
        mapping_type = type(value)
        return default_types.Dict({k: from_value(value[k], context) for k in value}, mapping_type)
    if util.is_attrs(value):
        return default_types.Attrs.from_type_and_attributes(type(value), tuple((from_value(getattr(value, a.name), context) for a in value.__attrs_attrs__)))
    if util.is_np_ndarray(value):
        ndarray = value.__array__()
        return default_types.TENSOR(ndarray.shape, ndarray.dtype)
    if isinstance(value, custom_nest_protocol.CustomNestProtocol):
        (metadata, components) = value.__tf_flatten__()
        return custom_nest_trace_type.CustomNestTraceType(type(value), metadata, tuple((from_value(c, context) for c in components)))
    try:
        ref = weakref.ref(value)
        if ref is None:
            raise TypeError(f'Deleted objects are not valid tf.function arguments, Got {value!r}')
        else:
            return default_types.Weakref(ref)
    except TypeError:
        try:
            return default_types.Literal(value)
        except:
            raise TypeError(f'Could not generate a generic TraceType for {value!r}.Please verify that it is immutable/hashable. Otheriwse, consider implementing the Tracing Protocol for it.')