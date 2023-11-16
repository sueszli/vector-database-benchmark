import array
import math
import struct
import pytest
from PIL import Image, ImagePath

def test_path():
    if False:
        for i in range(10):
            print('nop')
    p = ImagePath.Path(list(range(10)))
    assert len(p) == 5
    assert p[0] == (0.0, 1.0)
    assert p[-1] == (8.0, 9.0)
    assert list(p[:1]) == [(0.0, 1.0)]
    with pytest.raises(TypeError) as cm:
        p['foo']
    assert str(cm.value) == 'Path indices must be integers, not str'
    assert list(p) == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0), (8.0, 9.0)]
    assert p.tolist() == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0), (8.0, 9.0)]
    assert p.tolist(True) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert p.getbbox() == (0.0, 1.0, 8.0, 9.0)
    assert p.compact(5) == 2
    assert list(p) == [(0.0, 1.0), (4.0, 5.0), (8.0, 9.0)]
    p.transform((1, 0, 1, 0, 1, 1))
    assert list(p) == [(1.0, 2.0), (5.0, 6.0), (9.0, 10.0)]

@pytest.mark.parametrize('coords', ((0, 1), [0, 1], (0.0, 1.0), [0.0, 1.0], ((0, 1),), [(0, 1)], ((0.0, 1.0),), [(0.0, 1.0)], array.array('f', [0, 1]), array.array('f', [0, 1]).tobytes(), ImagePath.Path((0, 1))))
def test_path_constructors(coords):
    if False:
        for i in range(10):
            print('nop')
    p = ImagePath.Path(coords)
    assert list(p) == [(0.0, 1.0)]

@pytest.mark.parametrize('coords', (('a', 'b'), ([0, 1],), [[0, 1]], ([0.0, 1.0],), [[0.0, 1.0]]))
def test_invalid_path_constructors(coords):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as e:
        ImagePath.Path(coords)
    assert str(e.value) == 'incorrect coordinate type'

@pytest.mark.parametrize('coords', ((0,), [0], (0, 1, 2), [0, 1, 2]))
def test_path_odd_number_of_coordinates(coords):
    if False:
        print('Hello World!')
    with pytest.raises(ValueError) as e:
        ImagePath.Path(coords)
    assert str(e.value) == 'wrong number of coordinates'

@pytest.mark.parametrize('coords, expected', [([0, 1, 2, 3], (0.0, 1.0, 2.0, 3.0)), ([3, 2, 1, 0], (1.0, 0.0, 3.0, 2.0)), (0, (0.0, 0.0, 0.0, 0.0)), (1, (0.0, 0.0, 0.0, 0.0))])
def test_getbbox(coords, expected):
    if False:
        i = 10
        return i + 15
    p = ImagePath.Path(coords)
    assert p.getbbox() == expected

def test_getbbox_no_args():
    if False:
        return 10
    p = ImagePath.Path([0, 1, 2, 3])
    with pytest.raises(TypeError):
        p.getbbox(1)

@pytest.mark.parametrize('coords, expected', [(0, []), (list(range(6)), [(0.0, 3.0), (4.0, 9.0), (8.0, 15.0)])])
def test_map(coords, expected):
    if False:
        i = 10
        return i + 15
    p = ImagePath.Path(coords)
    p.map(lambda x, y: (x * 2, y * 3))
    assert list(p) == expected

def test_transform():
    if False:
        print('Hello World!')
    p = ImagePath.Path([0, 1, 2, 3])
    theta = math.pi / 15
    p.transform((math.cos(theta), math.sin(theta), 20, -math.sin(theta), math.cos(theta), 20))
    assert p.tolist() == [(20.20791169081776, 20.978147600733806), (22.58003027392089, 22.518619420565898)]

def test_transform_with_wrap():
    if False:
        while True:
            i = 10
    p = ImagePath.Path([0, 1, 2, 3])
    theta = math.pi / 15
    p.transform((math.cos(theta), math.sin(theta), 20, -math.sin(theta), math.cos(theta), 20), 1.0)
    assert p.tolist() == [(0.20791169081775962, 20.978147600733806), (0.5800302739208902, 22.518619420565898)]

def test_overflow_segfault():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises((TypeError, MemoryError)):
        x = Evil()
        for i in range(200000):
            x[i] = b'0' * 16

class Evil:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.corrupt = Image.core.path(4611686018427387904)

    def __getitem__(self, i):
        if False:
            while True:
                i = 10
        x = self.corrupt[i]
        return struct.pack('dd', x[0], x[1])

    def __setitem__(self, i, x):
        if False:
            i = 10
            return i + 15
        self.corrupt[i] = struct.unpack('dd', x)