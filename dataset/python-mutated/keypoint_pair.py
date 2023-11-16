from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union
import pyarrow as pa
from attrs import define, field
from .. import datatypes
from .._baseclasses import BaseBatch, BaseExtensionType
from .keypoint_pair_ext import KeypointPairExt
__all__ = ['KeypointPair', 'KeypointPairArrayLike', 'KeypointPairBatch', 'KeypointPairLike', 'KeypointPairType']

def _keypoint_pair__keypoint0__special_field_converter_override(x: datatypes.KeypointIdLike) -> datatypes.KeypointId:
    if False:
        return 10
    if isinstance(x, datatypes.KeypointId):
        return x
    else:
        return datatypes.KeypointId(x)

def _keypoint_pair__keypoint1__special_field_converter_override(x: datatypes.KeypointIdLike) -> datatypes.KeypointId:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(x, datatypes.KeypointId):
        return x
    else:
        return datatypes.KeypointId(x)

@define(init=False)
class KeypointPair(KeypointPairExt):
    """**Datatype**: A connection between two `Keypoints`."""

    def __init__(self: Any, keypoint0: datatypes.KeypointIdLike, keypoint1: datatypes.KeypointIdLike):
        if False:
            print('Hello World!')
        '\n        Create a new instance of the KeypointPair datatype.\n\n        Parameters\n        ----------\n        keypoint0:\n            The first point of the pair.\n        keypoint1:\n            The second point of the pair.\n        '
        self.__attrs_init__(keypoint0=keypoint0, keypoint1=keypoint1)
    keypoint0: datatypes.KeypointId = field(converter=_keypoint_pair__keypoint0__special_field_converter_override)
    keypoint1: datatypes.KeypointId = field(converter=_keypoint_pair__keypoint1__special_field_converter_override)
if TYPE_CHECKING:
    KeypointPairLike = Union[KeypointPair, Sequence[datatypes.KeypointIdLike]]
else:
    KeypointPairLike = Any
KeypointPairArrayLike = Union[KeypointPair, Sequence[KeypointPairLike]]

class KeypointPairType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.datatypes.KeypointPair'

    def __init__(self) -> None:
        if False:
            return 10
        pa.ExtensionType.__init__(self, pa.struct([pa.field('keypoint0', pa.uint16(), nullable=False, metadata={}), pa.field('keypoint1', pa.uint16(), nullable=False, metadata={})]), self._TYPE_NAME)

class KeypointPairBatch(BaseBatch[KeypointPairArrayLike]):
    _ARROW_TYPE = KeypointPairType()

    @staticmethod
    def _native_to_pa_array(data: KeypointPairArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            i = 10
            return i + 15
        return KeypointPairExt.native_to_pa_array_override(data, data_type)