import numpy as np
from skimage.util._map_array import map_array, ArrayMap
from skimage._shared import testing

def test_map_array_incorrect_output_shape():
    if False:
        for i in range(10):
            print('nop')
    labels = np.random.randint(0, 5, size=(24, 25))
    out = np.empty((24, 24))
    in_values = np.unique(labels)
    out_values = np.random.random(in_values.shape).astype(out.dtype)
    with testing.raises(ValueError):
        map_array(labels, in_values, out_values, out=out)

def test_map_array_non_contiguous_output_array():
    if False:
        i = 10
        return i + 15
    labels = np.random.randint(0, 5, size=(24, 25))
    out = np.empty((24 * 3, 25 * 2))[::3, ::2]
    in_values = np.unique(labels)
    out_values = np.random.random(in_values.shape).astype(out.dtype)
    with testing.raises(ValueError):
        map_array(labels, in_values, out_values, out=out)

def test_arraymap_long_str():
    if False:
        for i in range(10):
            print('nop')
    labels = np.random.randint(0, 40, size=(24, 25))
    in_values = np.unique(labels)
    out_values = np.random.random(in_values.shape)
    m = ArrayMap(in_values, out_values)
    assert len(str(m).split('\n')) == m._max_str_lines + 2

def test_arraymap_update():
    if False:
        i = 10
        return i + 15
    in_values = np.unique(np.random.randint(0, 200, size=5))
    out_values = np.random.random(len(in_values))
    m = ArrayMap(in_values, out_values)
    image = np.random.randint(1, len(m), size=(512, 512))
    assert np.all(m[image] < 1)
    m[1:] += 1
    assert np.all(m[image] >= 1)

def test_arraymap_bool_index():
    if False:
        return 10
    in_values = np.unique(np.random.randint(0, 200, size=5))
    out_values = np.random.random(len(in_values))
    m = ArrayMap(in_values, out_values)
    image = np.random.randint(1, len(in_values), size=(512, 512))
    assert np.all(m[image] < 1)
    positive = np.ones(len(m), dtype=bool)
    positive[0] = False
    m[positive] += 1
    assert np.all(m[image] >= 1)