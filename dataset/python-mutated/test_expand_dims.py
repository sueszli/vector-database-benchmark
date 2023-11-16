from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import numpy as np
from nose_utils import assert_raises

def get_data(shapes):
    if False:
        while True:
            i = 10
    return [np.empty(shape, dtype=np.uint8) for shape in shapes]

@pipeline_def
def expand_dims_pipe(shapes, axes=None, new_axis_names=None, layout=None):
    if False:
        i = 10
        return i + 15
    data = fn.external_source(lambda : get_data(shapes), layout=layout, batch=True, device='cpu')
    return fn.expand_dims(data, axes=axes, new_axis_names=new_axis_names)

def _testimpl_expand_dims(axes, new_axis_names, layout, shapes, expected_out_shapes, expected_layout):
    if False:
        i = 10
        return i + 15
    batch_size = len(shapes)
    pipe = expand_dims_pipe(batch_size=batch_size, num_threads=1, device_id=0, shapes=shapes, axes=axes, new_axis_names=new_axis_names, layout=layout)
    pipe.build()
    for _ in range(3):
        outs = pipe.run()
        assert outs[0].layout() == expected_layout
        for i in range(batch_size):
            out_arr = np.array(outs[0][i])
            assert out_arr.shape == expected_out_shapes[i]

def test_expand_dims():
    if False:
        for i in range(10):
            print('nop')
    args = [([0, 2], 'AB', 'XYZ', [(10, 20, 30)], [(1, 10, 1, 20, 30)], 'AXBYZ'), ([0, 3], None, 'XYZ', [(10, 20, 30)], [(1, 10, 20, 1, 30)], ''), ([3], None, 'XYZ', [(10, 20, 30), (100, 200, 300)], [(10, 20, 30, 1), (100, 200, 300, 1)], ''), ([4, 3], None, 'XYZ', [(10, 20, 30), (100, 200, 300)], [(10, 20, 30, 1, 1), (100, 200, 300, 1, 1)], ''), ([0, 1, 3, 5, 7], 'ABCDE', 'XYZ', [(11, 22, 33)], [(1, 1, 11, 1, 22, 1, 33, 1)], 'ABXCYDZE'), ([], '', 'HW', [(10, 20)], [(10, 20)], 'HW'), ([0, 1], '', '', [()], [(1, 1)], ''), ([0], '', 'HW', [(10, 20)], [(1, 10, 20)], ''), ([4, 3], 'AB', 'XYZ', [(10, 20, 30)], [(10, 20, 30, 1, 1)], 'XYZBA'), ([0], 'X', '', [()], [(1,)], 'X')]
    for (axes, new_axis_names, layout, shapes, expected_out_shapes, expected_layout) in args:
        yield (_testimpl_expand_dims, axes, new_axis_names, layout, shapes, expected_out_shapes, expected_layout)

def test_expand_dims_throw_error():
    if False:
        while True:
            i = 10
    args = [([4], None, None, [(10, 20, 30)], 'Data has not enough dimensions to add new axes at specified indices.'), ([0, -1], None, None, [(10, 20, 30)], "Axis value can't be negative"), ([2, 0, 2], 'AB', 'XYZ', [(10, 20, 30)], 'Specified [\\d]+ new dimensions, but layout contains only [\\d]+ new dimension names'), ([2], 'C', None, [(10, 20, 30)], 'Specifying ``new_axis_names`` requires an input with a proper layout.')]
    for (axes, new_axis_names, layout, shapes, err_msg) in args:
        pipe = expand_dims_pipe(batch_size=len(shapes), num_threads=1, device_id=0, shapes=shapes, axes=axes, new_axis_names=new_axis_names, layout=layout)
        with assert_raises(RuntimeError, regex=err_msg):
            pipe.build()
            pipe.run()