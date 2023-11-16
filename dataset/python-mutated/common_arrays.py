from __future__ import annotations
from typing import Any
import numpy as np
import torch
from rerun.components import ClassId, ClassIdBatch, Color, ColorBatch, DrawOrder, DrawOrderBatch, DrawOrderLike, InstanceKey, InstanceKeyBatch, KeypointId, KeypointIdBatch, Radius, RadiusArrayLike, RadiusBatch, TextBatch
from rerun.datatypes import Angle, Quaternion, Rgba32ArrayLike, Rotation3D, Rotation3DArrayLike, RotationAxisAngle, Utf8, Utf8ArrayLike, Vec2D, Vec2DArrayLike, Vec2DBatch, Vec3D, Vec3DArrayLike, Vec3DBatch, Vec4D, Vec4DArrayLike, Vec4DBatch
U64_MAX_MINUS_1 = 2 ** 64 - 2
U64_MAX = 2 ** 64 - 1

def none_empty_or_value(obj: Any, value: Any) -> Any:
    if False:
        while True:
            i = 10
    '\n    Helper function to make value align with None / Empty types.\n\n    If obj is None or an empty list, it is returned. Otherwise value\n    is returned. This is useful for creating the `_expected` functions.\n    '
    if obj is None:
        return None
    elif hasattr(obj, '__len__') and len(obj) == 0:
        return []
    else:
        return value
vec2ds_arrays: list[Vec2DArrayLike] = [[], np.array([]), [Vec2D([1, 2]), Vec2D([3, 4])], [np.array([1, 2], dtype=np.float32), np.array([3, 4], dtype=np.float32)], [(1, 2), (3, 4)], torch.tensor([(1, 2), (3, 4)], dtype=torch.float32), [1, 2, 3, 4], np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([1, 2, 3, 4], dtype=np.float32), np.array([1, 2, 3, 4], dtype=np.float32).reshape((2, 2, 1, 1, 1)), torch.asarray([1, 2, 3, 4], dtype=torch.float32)]

def vec2ds_expected(obj: Any, type_: Any | None=None) -> Any:
    if False:
        i = 10
        return i + 15
    if type_ is None:
        type_ = Vec2DBatch
    expected = none_empty_or_value(obj, [[1.0, 2.0], [3.0, 4.0]])
    return type_._optional(expected)
vec3ds_arrays: list[Vec3DArrayLike] = [[], np.array([]), [Vec3D([1, 2, 3]), Vec3D([4, 5, 6])], [np.array([1, 2, 3], dtype=np.float32), np.array([4, 5, 6], dtype=np.float32)], [(1, 2, 3), (4, 5, 6)], torch.tensor([(1, 2, 3), (4, 5, 6)], dtype=torch.float32), [1, 2, 3, 4, 5, 6], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), np.array([1, 2, 3, 4, 5, 6], dtype=np.float32), np.array([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape((2, 3, 1, 1, 1)), torch.asarray([1, 2, 3, 4, 5, 6], dtype=torch.float32)]

def vec3ds_expected(obj: Any, type_: Any | None=None) -> Any:
    if False:
        print('Hello World!')
    if type_ is None:
        type_ = Vec3DBatch
    expected = none_empty_or_value(obj, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return type_._optional(expected)
vec4ds_arrays: list[Vec4DArrayLike] = [[], np.array([]), [Vec4D([1, 2, 3, 4]), Vec4D([5, 6, 7, 8])], [np.array([1, 2, 3, 4], dtype=np.float32), np.array([5, 6, 7, 8], dtype=np.float32)], [(1, 2, 3, 4), (5, 6, 7, 8)], torch.tensor([(1, 2, 3, 4), (5, 6, 7, 8)], dtype=torch.float32), [1, 2, 3, 4, 5, 6, 7, 8], np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32), np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32), np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32).reshape((2, 4, 1, 1, 1)), torch.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)]

def vec4ds_expected(obj: Any, type_: Any | None=None) -> Any:
    if False:
        for i in range(10):
            print('nop')
    if type_ is None:
        type_ = Vec4DBatch
    expected = none_empty_or_value(obj, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    return type_._optional(expected)
rotations_arrays: list[Rotation3DArrayLike] = [[], Rotation3D(Quaternion(xyzw=[1, 2, 3, 4])), Rotation3D(Quaternion(xyzw=torch.tensor([1, 2, 3, 4]))), Rotation3D(RotationAxisAngle([1.0, 2.0, 3.0], Angle(4))), Quaternion(xyzw=[1, 2, 3, 4]), Quaternion(xyzw=[1.0, 2.0, 3.0, 4.0]), Quaternion(xyzw=np.array([1, 2, 3, 4])), Quaternion(xyzw=torch.tensor([1, 2, 3, 4])), RotationAxisAngle([1, 2, 3], 4), RotationAxisAngle([1.0, 2.0, 3.0], Angle(4)), RotationAxisAngle(Vec3D([1, 2, 3]), Angle(4)), RotationAxisAngle(np.array([1, 2, 3], dtype=np.uint8), Angle(rad=4)), RotationAxisAngle(torch.tensor([1, 2, 3]), Angle(rad=4)), [Rotation3D(Quaternion(xyzw=[1, 2, 3, 4])), [1, 2, 3, 4], Quaternion(xyzw=[1, 2, 3, 4]), RotationAxisAngle([1, 2, 3], 4)]]

def expected_rotations(rotations: Rotation3DArrayLike, type_: Any) -> Any:
    if False:
        while True:
            i = 10
    if rotations is None:
        return type_._optional(None)
    elif hasattr(rotations, '__len__') and len(rotations) == 0:
        return type_._optional(rotations)
    elif isinstance(rotations, Rotation3D):
        return type_._optional(rotations)
    elif isinstance(rotations, RotationAxisAngle):
        return type_._optional(RotationAxisAngle([1, 2, 3], 4))
    elif isinstance(rotations, Quaternion):
        return type_._optional(Quaternion(xyzw=[1, 2, 3, 4]))
    else:
        return type_._optional([Quaternion(xyzw=[1, 2, 3, 4])] * 3 + [RotationAxisAngle([1, 2, 3], 4)])
radii_arrays: list[RadiusArrayLike | None] = [None, [], np.array([]), [1, 10], [Radius(1), Radius(10)], np.array([1, 10], dtype=np.float32)]

def radii_expected(obj: Any) -> Any:
    if False:
        i = 10
        return i + 15
    expected = none_empty_or_value(obj, [1, 10])
    return RadiusBatch._optional(expected)
colors_arrays: list[Rgba32ArrayLike | None] = [None, [], np.array([]), [2852126924, 12255453], [Color(2852126924), Color(12255453)], np.array([[170, 0, 0, 204], [0, 187, 0, 221]], dtype=np.uint8), np.array([[2852126924], [12255453]], dtype=np.uint32), np.array([[170 / 255, 0.0, 0.0, 204 / 255], [0.0, 187 / 255, 0.0, 221 / 255]], dtype=np.float32), np.array([[170 / 255, 0.0, 0.0, 204 / 255], [0.0, 187 / 255, 0.0, 221 / 255]], dtype=np.float64), torch.tensor([[170 / 255, 0.0, 0.0, 204 / 255], [0.0, 187 / 255, 0.0, 221 / 255]], dtype=torch.float64), np.array([170, 0, 0, 204, 0, 187, 0, 221], dtype=np.uint8), np.array([2852126924, 12255453], dtype=np.uint32), np.array([170 / 255, 0.0, 0.0, 204 / 255, 0.0, 187 / 255, 0.0, 221 / 255], dtype=np.float32), np.array([170 / 255, 0.0, 0.0, 204 / 255, 0.0, 187 / 255, 0.0, 221 / 255], dtype=np.float64)]

def colors_expected(obj: Any) -> Any:
    if False:
        i = 10
        return i + 15
    expected = none_empty_or_value(obj, [2852126924, 12255453])
    return ColorBatch._optional(expected)
labels_arrays: list[Utf8ArrayLike | None] = [None, [], ['hello', 'friend'], [Utf8('hello'), Utf8('friend')]]

def labels_expected(obj: Any) -> Any:
    if False:
        print('Hello World!')
    expected = none_empty_or_value(obj, ['hello', 'friend'])
    return TextBatch._optional(expected)
draw_orders: list[DrawOrderLike | None] = [None, 300, DrawOrder(300)]

def draw_order_expected(obj: Any) -> Any:
    if False:
        print('Hello World!')
    expected = none_empty_or_value(obj, [300])
    return DrawOrderBatch._optional(expected)
class_ids_arrays = [[], np.array([]), [126, 127], [ClassId(126), ClassId(127)], np.array([126, 127], dtype=np.uint8), np.array([126, 127], dtype=np.uint16), np.array([126, 127], dtype=np.uint32), np.array([126, 127], dtype=np.uint64), torch.tensor([126, 127], dtype=torch.uint8)]

def class_ids_expected(obj: Any) -> Any:
    if False:
        i = 10
        return i + 15
    expected = none_empty_or_value(obj, [126, 127])
    return ClassIdBatch._optional(expected)
keypoint_ids_arrays = [[], np.array([]), [2, 3], [KeypointId(2), KeypointId(3)], np.array([2, 3], dtype=np.uint8), np.array([2, 3], dtype=np.uint16), np.array([2, 3], dtype=np.uint32), np.array([2, 3], dtype=np.uint64), torch.tensor([2, 3], dtype=torch.uint8)]

def keypoint_ids_expected(obj: Any) -> Any:
    if False:
        i = 10
        return i + 15
    expected = none_empty_or_value(obj, [2, 3])
    return KeypointIdBatch._optional(expected)
instance_keys_arrays = [[], np.array([]), [U64_MAX_MINUS_1, U64_MAX], [InstanceKey(U64_MAX_MINUS_1), InstanceKey(U64_MAX)], np.array([U64_MAX_MINUS_1, U64_MAX], dtype=np.uint64)]

def instance_keys_expected(obj: Any) -> Any:
    if False:
        print('Hello World!')
    expected = none_empty_or_value(obj, [U64_MAX_MINUS_1, U64_MAX])
    return InstanceKeyBatch._optional(expected)