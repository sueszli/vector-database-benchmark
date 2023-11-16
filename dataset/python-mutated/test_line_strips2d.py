from __future__ import annotations
import itertools
from typing import Any, Optional, cast
import numpy as np
import pytest
import rerun as rr
import torch
from rerun.components import DrawOrderLike, InstanceKeyArrayLike, LineStrip2DArrayLike, LineStrip2DBatch, RadiusArrayLike
from rerun.datatypes import ClassIdArrayLike, Rgba32ArrayLike, Utf8ArrayLike, Vec2D
from .common_arrays import class_ids_arrays, class_ids_expected, colors_arrays, colors_expected, draw_order_expected, draw_orders, instance_keys_arrays, instance_keys_expected, labels_arrays, labels_expected, none_empty_or_value, radii_arrays, radii_expected
strips_arrays: list[LineStrip2DArrayLike] = [[], np.array([]), [[[0, 0], [2, 1], [4, -1], [6, 0]], [[0, 3], [1, 4], [2, 2], [3, 4], [4, 2], [5, 4], [6, 3]]], [[Vec2D([0, 0]), (2, 1), [4, -1], (6, 0)], [Vec2D([0, 3]), (1, 4), [2, 2], (3, 4), [4, 2], (5, 4), [6, 3]]], [np.array([[0, 0], (2, 1), [4, -1], (6, 0)], dtype=np.float32), np.array([[0, 3], (1, 4), [2, 2], (3, 4), [4, 2], (5, 4), [6, 3]], dtype=np.float32)], [torch.tensor([[0, 0], (2, 1), [4, -1], (6, 0)], dtype=torch.float32), torch.tensor([[0, 3], (1, 4), [2, 2], (3, 4), [4, 2], (5, 4), [6, 3]], dtype=torch.float32)]]

def line_strips2d_expected(obj: Any) -> Any:
    if False:
        i = 10
        return i + 15
    expected = none_empty_or_value(obj, [[[0, 0], [2, 1], [4, -1], [6, 0]], [[0, 3], [1, 4], [2, 2], [3, 4], [4, 2], [5, 4], [6, 3]]])
    return LineStrip2DBatch(expected)

def test_line_strips2d() -> None:
    if False:
        print('Hello World!')
    all_arrays = itertools.zip_longest(strips_arrays, radii_arrays, colors_arrays, labels_arrays, draw_orders, class_ids_arrays, instance_keys_arrays)
    for (strips, radii, colors, labels, draw_order, class_ids, instance_keys) in all_arrays:
        strips = strips if strips is not None else strips_arrays[-1]
        strips = cast(LineStrip2DArrayLike, strips)
        radii = cast(Optional[RadiusArrayLike], radii)
        colors = cast(Optional[Rgba32ArrayLike], colors)
        labels = cast(Optional[Utf8ArrayLike], labels)
        draw_order = cast(Optional[DrawOrderLike], draw_order)
        class_ids = cast(Optional[ClassIdArrayLike], class_ids)
        instance_keys = cast(Optional[InstanceKeyArrayLike], instance_keys)
        print(f'rr.LineStrips2D(\n    {strips}\n    radii={radii!r}\n    colors={colors!r}\n    labels={labels!r}\n    draw_order={draw_order!r}\n    class_ids={class_ids!r}\n    instance_keys={instance_keys!r}\n)')
        arch = rr.LineStrips2D(strips, radii=radii, colors=colors, labels=labels, draw_order=draw_order, class_ids=class_ids, instance_keys=instance_keys)
        print(f'{arch}\n')
        assert arch.strips == line_strips2d_expected(strips)
        assert arch.radii == radii_expected(radii)
        assert arch.colors == colors_expected(colors)
        assert arch.labels == labels_expected(labels)
        assert arch.draw_order == draw_order_expected(draw_order)
        assert arch.class_ids == class_ids_expected(class_ids)
        assert arch.instance_keys == instance_keys_expected(instance_keys)

@pytest.mark.parametrize('data', [[[[0, 0], [2, 1]], [[4, -1], [6, 0]]], np.array([[0, 0], [2, 1], [4, -1], [6, 0]]).reshape([2, 2, 2])])
def test_line_segments2d(data: LineStrip2DArrayLike) -> None:
    if False:
        i = 10
        return i + 15
    arch = rr.LineStrips2D(data)
    assert arch.strips == LineStrip2DBatch([[[0, 0], [2, 1]], [[4, -1], [6, 0]]])

def test_single_line_strip2d() -> None:
    if False:
        print('Hello World!')
    reference = rr.LineStrips2D([rr.components.LineStrip2D([[0, 0], [1, 1]])])
    assert len(reference.strips) == 1
    assert reference == rr.LineStrips2D(rr.components.LineStrip2D([[0, 0], [1, 1]]))
    assert reference == rr.LineStrips2D([[[0, 0], [1, 1]]])
    assert reference == rr.LineStrips2D([[0, 0], [1, 1]])
    assert reference == rr.LineStrips2D(np.array([[0, 0], [1, 1]]))
    assert reference == rr.LineStrips2D([np.array([0, 0]), np.array([1, 1])])

def test_line_strip2d_invalid_shapes() -> None:
    if False:
        for i in range(10):
            print('nop')
    rr.set_strict_mode(True)
    with pytest.raises(ValueError):
        rr.LineStrips2D([[0, 0, 2, 1, 4, -1, 6, 0], [0, 3, 1, 4, 2, 2, 3, 4, 4, 2, 5, 4, 6, 3]])
    with pytest.raises(ValueError):
        rr.LineStrips2D([np.array([0, 0, 2, 1, 4, -1, 6, 0], dtype=np.float32), np.array([0, 3, 1, 4, 2, 2, 3, 4, 4, 2, 5, 4, 6, 3], dtype=np.float32)])
    with pytest.raises(ValueError):
        rr.LineStrips2D(np.array([[[0, 0], (2, 1), [4, -1], (6, 0)], [[0, 3], (1, 4), [2, 2], (3, 4), [4, 2], (5, 4), [6, 3]]]))
    with pytest.raises(ValueError):
        rr.LineStrips2D(np.array([[0, 0, 2, 1, 4, -1, 6, 0], [0, 3, 1, 4, 2, 2, 3, 4, 4, 2, 5, 4, 6, 3]]))
if __name__ == '__main__':
    test_line_strips2d()