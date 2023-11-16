from __future__ import annotations
from typing import TYPE_CHECKING
from ..error_utils import _send_warning_or_raise, catch_and_log_exceptions
if TYPE_CHECKING:
    from ..components import TensorDataBatch
    from ..datatypes import TensorDataArrayLike

class BarChartExt:
    """Extension for [BarChart][rerun.archetypes.BarChart]."""

    @staticmethod
    @catch_and_log_exceptions('BarChart converter')
    def values__field_converter_override(data: TensorDataArrayLike) -> TensorDataBatch:
        if False:
            return 10
        from ..components import TensorDataBatch
        tensor_data = TensorDataBatch(data)
        shape_dims = tensor_data.as_arrow_array()[0].value['shape'].values.field(0).to_numpy()
        if len(shape_dims) != 1:
            _send_warning_or_raise(f'Bar chart data should only be 1D. Got values with shape: {shape_dims}', 2, recording=None)
        return tensor_data