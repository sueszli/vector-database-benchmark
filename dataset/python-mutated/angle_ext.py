from __future__ import annotations
from typing import TYPE_CHECKING, Any
import pyarrow as pa
if TYPE_CHECKING:
    from . import AngleArrayLike

class AngleExt:
    """Extension for [Angle][rerun.datatypes.Angle]."""

    def __init__(self: Any, rad: float | None=None, deg: float | None=None) -> None:
        if False:
            return 10
        '\n        Create a new instance of the Angle datatype.\n\n        Parameters\n        ----------\n        rad:\n            Angle in radians, specify either `rad` or `deg`.\n        deg:\n            Angle in degrees, specify either `rad` or `deg`.\n        '
        if rad is not None:
            self.__attrs_init__(inner=rad, kind='radians')
        elif deg is not None:
            self.__attrs_init__(inner=deg, kind='degrees')
        else:
            raise ValueError('Either `rad` or `deg` must be provided.')

    @staticmethod
    def native_to_pa_array_override(data: AngleArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            for i in range(10):
                print('nop')
        from . import Angle
        if isinstance(data, Angle) or isinstance(data, float):
            data = [data]
        types: list[int] = []
        value_offsets: list[int] = []
        num_nulls = 0
        radians: list[float] = []
        degrees: list[float] = []
        null_type_idx = 0
        radian_type_idx = 1
        degree_type_idx = 2
        for angle in data:
            if angle is None:
                value_offsets.append(num_nulls)
                num_nulls += 1
                types.append(null_type_idx)
            else:
                if isinstance(angle, float):
                    angle = Angle(angle)
                if angle.kind == 'radians':
                    value_offsets.append(len(radians))
                    radians.append(angle.inner)
                    types.append(radian_type_idx)
                elif angle.kind == 'degrees':
                    value_offsets.append(len(degrees))
                    degrees.append(angle.inner)
                    types.append(degree_type_idx)
                else:
                    raise ValueError(f'Unknown angle kind: {angle.kind} (expected `radians` or `degrees`)')
        return pa.UnionArray.from_buffers(type=data_type, length=len(data), buffers=[None, pa.array(types, type=pa.int8()).buffers()[1], pa.array(value_offsets, type=pa.int32()).buffers()[1]], children=[pa.nulls(num_nulls, pa.null()), pa.array(radians, type=pa.float32()), pa.array(degrees, type=pa.float32())])