from __future__ import annotations
import itertools
from typing import Optional, cast
import numpy as np
import pytest
import rerun as rr
from rerun.components import Color, ColorBatch, DrawOrderLike, InstanceKeyArrayLike, Position2DBatch, RadiusArrayLike
from rerun.datatypes import ClassIdArrayLike, KeypointIdArrayLike, Rgba32ArrayLike, Utf8ArrayLike, Vec2DArrayLike
from .common_arrays import class_ids_arrays, class_ids_expected, colors_arrays, colors_expected, draw_order_expected, draw_orders, instance_keys_arrays, instance_keys_expected, keypoint_ids_arrays, keypoint_ids_expected, labels_arrays, labels_expected, radii_arrays, radii_expected
from .common_arrays import vec2ds_arrays as positions_arrays
from .common_arrays import vec2ds_expected as positions_expected

def test_points2d() -> None:
    if False:
        while True:
            i = 10
    all_arrays = itertools.zip_longest(positions_arrays, radii_arrays, colors_arrays, labels_arrays, draw_orders, class_ids_arrays, keypoint_ids_arrays, instance_keys_arrays)
    for (positions, radii, colors, labels, draw_order, class_ids, keypoint_ids, instance_keys) in all_arrays:
        positions = positions if positions is not None else positions_arrays[-1]
        positions = cast(Vec2DArrayLike, positions)
        radii = cast(Optional[RadiusArrayLike], radii)
        colors = cast(Optional[Rgba32ArrayLike], colors)
        labels = cast(Optional[Utf8ArrayLike], labels)
        draw_order = cast(Optional[DrawOrderLike], draw_order)
        class_ids = cast(Optional[ClassIdArrayLike], class_ids)
        keypoint_ids = cast(Optional[KeypointIdArrayLike], keypoint_ids)
        instance_keys = cast(Optional[InstanceKeyArrayLike], instance_keys)
        print(f'rr.Points2D(\n    {positions}\n    radii={radii!r}\n    colors={colors!r}\n    labels={labels!r}\n    draw_order={draw_order!r}\n    class_ids={class_ids!r}\n    keypoint_ids={keypoint_ids!r}\n    instance_keys={instance_keys!r}\n)')
        arch = rr.Points2D(positions, radii=radii, colors=colors, labels=labels, draw_order=draw_order, class_ids=class_ids, keypoint_ids=keypoint_ids, instance_keys=instance_keys)
        print(f'{arch}\n')
        assert arch.positions == positions_expected(positions, Position2DBatch)
        assert arch.radii == radii_expected(radii)
        assert arch.colors == colors_expected(colors)
        assert arch.labels == labels_expected(labels)
        assert arch.draw_order == draw_order_expected(draw_order)
        assert arch.class_ids == class_ids_expected(class_ids)
        assert arch.keypoint_ids == keypoint_ids_expected(keypoint_ids)
        assert arch.instance_keys == instance_keys_expected(instance_keys)

@pytest.mark.parametrize('data', [[0, 128, 0, 255], [0, 128, 0], np.array((0, 128, 0, 255)), [0.0, 0.5, 0.0, 1.0], np.array((0.0, 0.5, 0.0, 1.0))])
def test_point2d_single_color(data: Rgba32ArrayLike) -> None:
    if False:
        print('Hello World!')
    pts = rr.Points2D(positions=np.zeros((5, 2)), colors=data)
    assert pts.colors == ColorBatch(Color([0, 128, 0, 255]))

@pytest.mark.parametrize('data', [[[0, 128, 0, 255], [128, 0, 0, 255]], [[0, 128, 0], [128, 0, 0]], np.array([[0, 128, 0, 255], [128, 0, 0, 255]]), np.array([0, 128, 0, 255, 128, 0, 0, 255], dtype=np.uint8), np.array([8388863, 2147483903], dtype=np.uint32), np.array([[0, 128, 0], [128, 0, 0]]), [[0.0, 0.5, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]], [[0.0, 0.5, 0.0], [0.5, 0.0, 0.0]], np.array([[0.0, 0.5, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]]), np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0]]), np.array([0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0]), [8388863, 2147483903]])
def test_point2d_multiple_colors(data: Rgba32ArrayLike) -> None:
    if False:
        i = 10
        return i + 15
    pts = rr.Points2D(positions=np.zeros((5, 2)), colors=data)
    assert pts.colors == ColorBatch([Color([0, 128, 0, 255]), Color([128, 0, 0, 255])])
if __name__ == '__main__':
    test_points2d()