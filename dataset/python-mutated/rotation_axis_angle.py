from __future__ import annotations
from typing import Sequence, Union
import pyarrow as pa
from attrs import define, field
from .. import datatypes
from .._baseclasses import BaseBatch, BaseExtensionType
from .rotation_axis_angle_ext import RotationAxisAngleExt
__all__ = ['RotationAxisAngle', 'RotationAxisAngleArrayLike', 'RotationAxisAngleBatch', 'RotationAxisAngleLike', 'RotationAxisAngleType']

def _rotation_axis_angle__axis__special_field_converter_override(x: datatypes.Vec3DLike) -> datatypes.Vec3D:
    if False:
        while True:
            i = 10
    if isinstance(x, datatypes.Vec3D):
        return x
    else:
        return datatypes.Vec3D(x)

@define(init=False)
class RotationAxisAngle(RotationAxisAngleExt):
    """**Datatype**: 3D rotation represented by a rotation around a given axis."""
    axis: datatypes.Vec3D = field(converter=_rotation_axis_angle__axis__special_field_converter_override)
    angle: datatypes.Angle = field(converter=RotationAxisAngleExt.angle__field_converter_override)
RotationAxisAngleLike = RotationAxisAngle
RotationAxisAngleArrayLike = Union[RotationAxisAngle, Sequence[RotationAxisAngleLike]]

class RotationAxisAngleType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.datatypes.RotationAxisAngle'

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pa.ExtensionType.__init__(self, pa.struct([pa.field('axis', pa.list_(pa.field('item', pa.float32(), nullable=False, metadata={}), 3), nullable=False, metadata={}), pa.field('angle', pa.dense_union([pa.field('_null_markers', pa.null(), nullable=True, metadata={}), pa.field('Radians', pa.float32(), nullable=False, metadata={}), pa.field('Degrees', pa.float32(), nullable=False, metadata={})]), nullable=False, metadata={})]), self._TYPE_NAME)

class RotationAxisAngleBatch(BaseBatch[RotationAxisAngleArrayLike]):
    _ARROW_TYPE = RotationAxisAngleType()

    @staticmethod
    def _native_to_pa_array(data: RotationAxisAngleArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            for i in range(10):
                print('nop')
        return RotationAxisAngleExt.native_to_pa_array_override(data, data_type)