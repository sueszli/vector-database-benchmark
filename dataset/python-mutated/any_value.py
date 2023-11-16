from __future__ import annotations
from typing import Any, Iterable
import numpy as np
import pyarrow as pa
from ._log import AsComponents, ComponentBatchLike
from .error_utils import catch_and_log_exceptions
ANY_VALUE_TYPE_REGISTRY: dict[str, Any] = {}

class AnyBatchValue(ComponentBatchLike):
    """
    Helper to log arbitrary data as a component batch.

    This is a very simple helper that implements the `ComponentBatchLike` interface on top
    of the `pyarrow` library array conversion functions.

    See also [rerun.AnyValues][].
    """

    def __init__(self, name: str, value: Any, drop_untyped_nones: bool=True) -> None:
        if False:
            while True:
                i = 10
        "\n        Construct a new AnyBatchValue.\n\n        The value will be attempted to be converted into an arrow array by first calling\n        the `as_arrow_array()` method if it's defined. All Rerun Batch datatypes implement\n        this function so it's possible to pass them directly to AnyValues.\n\n        If the object doesn't implement `as_arrow_array()`, it will be passed as an argument\n        to [pyarrow.array][] .\n\n        Note: rerun requires that a given component only take on a single type.\n        The first type logged will be the type that is used for all future logs\n        of that component. The API will make a best effort to do type conversion\n        if supported by numpy and arrow. Any components that can't be converted\n        will be dropped, and a warning will be sent to the log.\n\n        If you are want to inspect how your component will be converted to the\n        underlying arrow code, the following snippet is what is happening\n        internally:\n\n        ```\n        np_value = np.atleast_1d(np.array(value, copy=False))\n        pa_value = pa.array(value)\n        ```\n\n        Parameters\n        ----------\n        name:\n            The name of the component.\n        value:\n            The data to be logged as a component.\n        drop_untyped_nones:\n            If True, any components that are None will be dropped unless they have been\n            previously logged with a type.\n        "
        (np_type, pa_type) = ANY_VALUE_TYPE_REGISTRY.get(name, (None, None))
        self.name = name
        self.pa_array = None
        with catch_and_log_exceptions(f"Converting data for '{name}'"):
            if isinstance(value, pa.Array):
                self.pa_array = value
            elif hasattr(value, 'as_arrow_array'):
                self.pa_array = value.as_arrow_array()
            elif np_type is not None:
                if value is None:
                    value = []
                np_value = np.atleast_1d(np.array(value, copy=False, dtype=np_type))
                self.pa_array = pa.array(np_value, type=pa_type)
            elif value is None:
                if not drop_untyped_nones:
                    raise ValueError('Cannot convert None to arrow array. Type is unknown.')
            else:
                np_value = np.atleast_1d(np.array(value, copy=False))
                self.pa_array = pa.array(np_value)
                ANY_VALUE_TYPE_REGISTRY[name] = (np_value.dtype, self.pa_array.type)

    def is_valid(self) -> bool:
        if False:
            return 10
        return self.pa_array is not None

    def component_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.name

    def as_arrow_array(self) -> pa.Array | None:
        if False:
            print('Hello World!')
        return self.pa_array

class AnyValues(AsComponents):
    """
    Helper to log arbitrary values as a bundle of components.

    Example
    -------
    ```python
    rr.log(
        "any_values", rr.AnyValues(
            foo=[1.2, 3.4, 5.6], bar="hello world",
        ),
    )
    ```
    """

    def __init__(self, drop_untyped_nones: bool=True, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        "\n        Construct a new AnyValues bundle.\n\n        Each kwarg will be logged as a separate component using the provided data.\n         - The key will be used as the name of the component\n         - The value must be able to be converted to an array of arrow types. In\n           general, if you can pass it to [pyarrow.array][] you can log it as a\n           extension component.\n\n        All values must either have the same length, or be singular in which\n        case they will be treated as a splat.\n\n        Note: rerun requires that a given component only take on a single type.\n        The first type logged will be the type that is used for all future logs\n        of that component. The API will make a best effort to do type conversion\n        if supported by numpy and arrow. Any components that can't be converted\n        will result in a warning (or an exception in strict mode).\n\n        `None` values provide a particular challenge as they have no type\n        information until after the component has been logged with a particular\n        type. By default, these values are dropped. This should generally be\n        fine as logging `None` to clear the value before it has been logged is\n        meaningless unless you are logging out-of-order data. In such cases,\n        consider introducing your own typed component via\n        [rerun.ComponentBatchLike][].\n\n        You can change this behavior by setting `drop_untyped_nones` to `False`,\n        but be aware that this will result in potential warnings (or exceptions\n        in strict mode).\n\n        If you are want to inspect how your component will be converted to the\n        underlying arrow code, the following snippet is what is happening\n        internally:\n        ```\n        np_value = np.atleast_1d(np.array(value, copy=False))\n        pa_value = pa.array(value)\n        ```\n\n        Parameters\n        ----------\n        drop_untyped_nones:\n            If True, any components that are None will be dropped unless they\n            have been previously logged with a type.\n        kwargs:\n            The components to be logged.\n\n        "
        global ANY_VALUE_TYPE_REGISTRY
        self.component_batches = []
        with catch_and_log_exceptions(self.__class__.__name__):
            for (name, value) in kwargs.items():
                batch = AnyBatchValue(name, value, drop_untyped_nones=drop_untyped_nones)
                if batch.is_valid():
                    self.component_batches.append(batch)

    def as_component_batches(self) -> Iterable[ComponentBatchLike]:
        if False:
            for i in range(10):
                print('nop')
        return self.component_batches