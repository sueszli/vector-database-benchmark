from __future__ import annotations
from typing import Any, Generic, Iterable, Protocol, TypeVar
import pyarrow as pa
from attrs import define, fields
from .error_utils import catch_and_log_exceptions
T = TypeVar('T')

class ComponentBatchLike(Protocol):
    """Describes interface for objects that can be converted to batch of rerun Components."""

    def component_name(self) -> str:
        if False:
            i = 10
            return i + 15
        'Returns the name of the component.'
        ...

    def as_arrow_array(self) -> pa.Array:
        if False:
            i = 10
            return i + 15
        '\n        Returns a `pyarrow.Array` of the component data.\n\n        Each element in the array corresponds to an instance of the component. Single-instanced\n        components and splats must still be represented as a 1-element array.\n        '
        ...

class AsComponents(Protocol):
    """
    Describes interface for interpreting an object as a bundle of Components.

    Note: the `num_instances()` function is an optional part of this interface. The method does not need to be
    implemented as it is only used after checking for its existence. (There is unfortunately no way to express this
    correctly with the Python typing system, see https://github.com/python/typing/issues/601).
    """

    def as_component_batches(self) -> Iterable[ComponentBatchLike]:
        if False:
            i = 10
            return i + 15
        '\n        Returns an iterable of `ComponentBatchLike` objects.\n\n        Each object in the iterable must adhere to the `ComponentBatchLike`\n        interface. All of the batches should have the same length as the value\n        returned by `num_instances`, or length 1 if the component is a splat.,\n        or 0 if the component is being cleared.\n        '
        ...

@define
class Archetype:
    """Base class for all archetypes."""

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        cls = type(self)
        s = f'rr.{cls.__name__}(\n'
        for fld in fields(cls):
            if 'component' in fld.metadata:
                comp = getattr(self, fld.name)
                datatype = getattr(comp, 'type', None)
                if datatype:
                    s += f'  {datatype.extension_name}<{datatype.storage_type}>(\n    {comp.to_pylist()}\n  )\n'
        s += ')'
        return s

    @classmethod
    def archetype_name(cls) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'rerun.archetypes.' + cls.__name__

    @classmethod
    def indicator(cls) -> ComponentBatchLike:
        if False:
            print('Hello World!')
        "\n        Creates a `ComponentBatchLike` out of the associated indicator component.\n\n        This allows for associating arbitrary indicator components with arbitrary data.\n        Check out the `manual_indicator` API example to see what's possible.\n        "
        from ._log import IndicatorComponentBatch
        return IndicatorComponentBatch(cls.archetype_name())

    def num_instances(self) -> int:
        if False:
            print('Hello World!')
        '\n        The number of instances that make up the batch.\n\n        Part of the `AsComponents` logging interface.\n        '
        for fld in fields(type(self)):
            if 'component' in fld.metadata and fld.metadata['component'] == 'required':
                return len(getattr(self, fld.name))
        raise ValueError('Archetype has no required components')

    def as_component_batches(self) -> Iterable[ComponentBatchLike]:
        if False:
            print('Hello World!')
        '\n        Return all the component batches that make up the archetype.\n\n        Part of the `AsComponents` logging interface.\n        '
        yield self.indicator()
        for fld in fields(type(self)):
            if 'component' in fld.metadata:
                comp = getattr(self, fld.name)
                if comp is not None:
                    yield comp
    __repr__ = __str__

class BaseExtensionType(pa.ExtensionType):
    """Extension type for datatypes and non-delegating components."""
    _TYPE_NAME: str
    'The name used when constructing the extension type.\n\n    Should following rerun typing conventions:\n     - `rerun.datatypes.<TYPE>` for datatypes\n     - `rerun.components.<TYPE>` for components\n\n    Many component types simply subclass a datatype type and override\n    the `_TYPE_NAME` field.\n    '
    _ARRAY_TYPE: type[pa.ExtensionArray] = pa.ExtensionArray
    'The extension array class associated with this class.'

    def __arrow_ext_serialize__(self) -> bytes:
        if False:
            return 10
        return b''

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type: Any, serialized: Any) -> pa.ExtensionType:
        if False:
            for i in range(10):
                print('nop')
        return cls()

class BaseBatch(Generic[T]):
    _ARROW_TYPE: BaseExtensionType = None
    'The pyarrow type of this batch.'

    def __init__(self, data: T | None, strict: bool | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new batch.\n\n        This method must flexibly accept native data (which comply with type `T`). Subclasses must provide a type\n        parameter specifying the type of the native data (this is automatically handled by the code generator).\n\n        A value of None indicates that the component should be cleared and results in the creation of an empty\n        array.\n\n        The actual creation of the Arrow array is delegated to the `_native_to_pa_array()` method, which is not\n        implemented by default.\n\n        Parameters\n        ----------\n        data : T | None\n            The data to convert into an Arrow array.\n        strict : bool | None\n            Whether to raise an exception if the data cannot be converted into an Arrow array. If None, the value\n            defaults to the value of the `rerun.strict` global setting.\n\n        Returns\n        -------\n        The Arrow array encapsulating the data.\n        '
        if data is not None:
            with catch_and_log_exceptions(self.__class__.__name__, strict=strict):
                if isinstance(data, pa.Array) and data.type == self._ARROW_TYPE:
                    self.pa_array = data
                elif isinstance(data, pa.Array) and data.type == self._ARROW_TYPE.storage_type:
                    self.pa_array = self._ARROW_TYPE.wrap_array(data)
                else:
                    self.pa_array = self._ARROW_TYPE.wrap_array(self._native_to_pa_array(data, self._ARROW_TYPE.storage_type))
                return
        self.pa_array = _empty_pa_array(self._ARROW_TYPE)

    @classmethod
    def _required(cls, data: T | None) -> BaseBatch[T]:
        if False:
            while True:
                i = 10
        '\n        Primary method for creating Arrow arrays for optional components.\n\n        Just calls through to __init__, but with clearer type annotations.\n        '
        return cls(data)

    @classmethod
    def _optional(cls, data: T | None) -> BaseBatch[T] | None:
        if False:
            return 10
        '\n        Primary method for creating Arrow arrays for optional components.\n\n        For optional components, the default value of None is preserved in the field to indicate that the optional\n        field was not specified.\n        If any value other than None is provided, it is passed through to `__init__`.\n\n        Parameters\n        ----------\n        data : T | None\n            The data to convert into an Arrow array.\n\n        Returns\n        -------\n        The Arrow array encapsulating the data.\n        '
        if data is None:
            return None
        else:
            return cls(data)

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, BaseBatch):
            return NotImplemented
        return self.pa_array == other.pa_array

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.pa_array)

    @staticmethod
    def _native_to_pa_array(data: T, data_type: pa.DataType) -> pa.Array:
        if False:
            return 10
        "\n        Converts native data into an Arrow array.\n\n        Subclasses must provide an implementation of this method (via an override) if they are to be used as either\n        an archetype's field (which should be the case for all components), or a (delegating) component's field (for\n        datatypes). Datatypes which are used only within other datatypes may omit implementing this method, provided\n        that the top-level datatype implements it.\n\n        A hand-coded override must be provided for the code generator to implement this method. The override must be\n        named `native_to_pa_array_override()` and exist as a static member of the `<TYPE>Ext` class located in\n        `<type>_ext.py`.\n\n        `ColorExt.native_to_pa_array_override()` in `color_ext.py` is a good example of how to implement this method, in\n        conjunction with the native type's converter (see `rgba__field_converter_override()`, used to construct the\n        native `Color` object).\n\n        Parameters\n        ----------\n        data : T\n            The data to convert into an Arrow array.\n        data_type : pa.DataType\n            The Arrow data type of the data.\n\n        Returns\n        -------\n        The Arrow array encapsulating the data.\n        "
        raise NotImplementedError

    def as_arrow_array(self) -> pa.Array:
        if False:
            i = 10
            return i + 15
        '\n        The component as an arrow batch.\n\n        Part of the `ComponentBatchLike` logging interface.\n        '
        return self.pa_array

class ComponentBatchMixin(ComponentBatchLike):

    def component_name(self) -> str:
        if False:
            while True:
                i = 10
        '\n        The name of the component.\n\n        Part of the `ComponentBatchLike` logging interface.\n        '
        return self._ARROW_TYPE._TYPE_NAME

@catch_and_log_exceptions(context='creating empty array')
def _empty_pa_array(type: pa.DataType) -> pa.Array:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(type, pa.ExtensionType):
        return type.wrap_array(_empty_pa_array(type.storage_type))
    if isinstance(type, pa.UnionType):
        return pa.UnionArray.from_buffers(type=type, length=0, buffers=[None, pa.array([], type=pa.int8()).buffers()[1], pa.array([], type=pa.int32()).buffers()[1]], children=[_empty_pa_array(field_type.type) for field_type in type])
    if isinstance(type, pa.StructType):
        return pa.StructArray.from_arrays([_empty_pa_array(field_type.type) for field_type in type], fields=list(type))
    return pa.array([], type=type)