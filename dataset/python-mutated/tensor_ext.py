from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence
from ..error_utils import catch_and_log_exceptions
if TYPE_CHECKING:
    from ..datatypes import TensorDataLike
    from ..datatypes.tensor_data_ext import TensorLike

class TensorExt:
    """Extension for [Tensor][rerun.archetypes.Tensor]."""

    def __init__(self: Any, data: TensorDataLike | TensorLike | None=None, *, dim_names: Sequence[str | None] | None=None):
        if False:
            i = 10
            return i + 15
        '\n        Construct a `Tensor` archetype.\n\n        The `Tensor` archetype internally contains a single component: `TensorData`.\n\n        See the `TensorData` constructor for more advanced options to interpret buffers\n        as `TensorData` of varying shapes.\n\n        For simple cases, you can pass array objects and optionally specify the names of\n        the dimensions. The shape of the `TensorData` will be inferred from the array.\n\n        Parameters\n        ----------\n        self:\n            The TensorData object to construct.\n        data: TensorDataLike | None\n            A TensorData object, or type that can be converted to a numpy array.\n        dim_names: Sequence[str] | None\n            The names of the tensor dimensions when generating the shape from an array.\n        '
        from ..datatypes import TensorData
        with catch_and_log_exceptions(context=self.__class__.__name__):
            if not isinstance(data, TensorData):
                data = TensorData(array=data, dim_names=dim_names)
            elif dim_names is not None:
                data = TensorData(buffer=data.buffer, dim_names=dim_names)
            self.__attrs_init__(data=data)
            return
        self.__attrs_clear__()