from panda3d import core
import pickle

def test_sparse_array_type():
    if False:
        return 10
    assert core.SparseArray.get_class_type().name == 'SparseArray'

def test_sparse_array_set_bit_to():
    if False:
        while True:
            i = 10
    'Tests SparseArray behavior for set_bit_to().'
    s = core.SparseArray()
    s.set_bit_to(5, True)
    assert s.get_bit(5)
    s = core.SparseArray.all_on()
    s.set_bit_to(5, False)
    assert not s.get_bit(5)

def test_sparse_array_clear():
    if False:
        return 10
    'Tests SparseArray behavior for clear().'
    s = core.SparseArray.all_on()
    s.clear()
    assert s.is_zero()
    assert not s.is_inverse()
    assert s.get_num_subranges() == 0
    assert s.get_num_on_bits() == 0
    assert s.get_num_bits() == 0
    s = core.SparseArray()
    s.set_range(5, 10)
    s.clear()
    assert s.is_zero()
    assert not s.is_inverse()
    assert s.get_num_subranges() == 0
    assert s.get_num_on_bits() == 0
    assert s.get_num_bits() == 0

def test_sparse_array_clear_range():
    if False:
        print('Hello World!')
    for mask in range(127):
        for begin in range(8):
            for size in range(8):
                b = core.BitArray(mask)
                s = core.SparseArray(b)
                s.clear_range(begin, size)
                b.clear_range(begin, size)
                assert core.BitArray(s) == b
                assert s == core.SparseArray(b)

def test_sparse_array_set_clear_ranges():
    if False:
        print('Hello World!')
    'Tests SparseArray behavior for setting and clearing ranges.'
    s = core.SparseArray()
    s.set_range(2, 3)
    s.clear_range(3, 3)
    assert s.get_bit(2)
    assert not s.get_bit(3)
    s = core.SparseArray()
    s.set_range_to(True, 2, 3)
    s.set_range_to(False, 3, 3)
    assert s.get_bit(2)
    assert not s.get_bit(3)
    s = core.SparseArray()
    s.set_range(2, 3)
    s.set_range(7, 3)
    s.clear_range(3, 6)
    assert s.get_bit(2)
    assert not s.get_bit(3)
    assert not s.get_bit(8)
    assert s.get_bit(9)
    s = core.SparseArray()
    s.set_range_to(True, 2, 3)
    s.set_range_to(True, 7, 3)
    s.set_range_to(False, 3, 6)
    assert s.get_bit(2)
    assert not s.get_bit(3)
    assert not s.get_bit(8)
    assert s.get_bit(9)

def test_sparse_array_set_range():
    if False:
        return 10
    'Tests SparseArray behavior for set_range().'
    s = core.SparseArray.all_on()
    s.clear_range(2, 3)
    s.set_range(3, 3)
    assert not s.get_bit(2)
    assert s.get_bit(3)
    s = core.SparseArray.all_on()
    s.set_range_to(False, 2, 3)
    s.set_range_to(True, 3, 3)
    assert not s.get_bit(2)
    assert s.get_bit(3)
    s = core.SparseArray.all_on()
    s.clear_range(2, 3)
    s.clear_range(7, 3)
    s.set_range(3, 6)
    assert not s.get_bit(2)
    assert s.get_bit(3)
    assert s.get_bit(8)
    assert not s.get_bit(9)
    s = core.SparseArray.all_on()
    s.set_range_to(False, 2, 3)
    s.set_range_to(False, 7, 3)
    s.set_range_to(True, 3, 6)
    assert not s.get_bit(2)
    assert s.get_bit(3)
    assert s.get_bit(8)
    assert not s.get_bit(9)

def test_sparse_array_bits_in_common():
    if False:
        i = 10
        return i + 15
    'Tests SparseArray behavior for has_bits_in_common().'
    s = core.SparseArray()
    t = core.SparseArray()
    s.set_range(2, 4)
    t.set_range(5, 4)
    assert s.has_bits_in_common(t)
    s = core.SparseArray()
    t = core.SparseArray()
    s.set_range(2, 4)
    t.set_range(6, 4)
    assert not s.has_bits_in_common(t)

def test_sparse_array_operations():
    if False:
        while True:
            i = 10
    'Tests SparseArray behavior for various operations.'
    s = core.SparseArray()
    s.set_bit(2)
    t = s << 2
    assert t.get_bit(4)
    assert not t.get_bit(2)
    s = core.SparseArray()
    s.set_bit(4)
    t = s >> 2
    assert t.get_bit(2)
    assert not t.get_bit(4)
    s = core.SparseArray()
    t = core.SparseArray()
    s.set_bit(2)
    s.set_bit(3)
    t.set_bit(1)
    t.set_bit(3)
    u = s & t
    assert not u.get_bit(0)
    assert not u.get_bit(1)
    assert not u.get_bit(2)
    assert u.get_bit(3)
    s = core.SparseArray()
    t = core.SparseArray()
    s.set_bit(2)
    s.set_bit(3)
    t.set_bit(1)
    t.set_bit(3)
    u = s | t
    assert not u.get_bit(0)
    assert u.get_bit(1)
    assert u.get_bit(2)
    assert u.get_bit(3)
    s = core.SparseArray()
    t = core.SparseArray()
    s.set_bit(2)
    s.set_bit(3)
    t.set_bit(1)
    t.set_bit(3)
    u = s ^ t
    assert not u.get_bit(0)
    assert u.get_bit(1)
    assert u.get_bit(2)
    assert not u.get_bit(3)

def test_sparse_array_augm_assignment():
    if False:
        i = 10
        return i + 15
    'Tests SparseArray behavior for augmented assignments.'
    s = t = core.SparseArray()
    t <<= 2
    assert s is t
    s = t = core.SparseArray()
    t >>= 2
    assert s is t
    s = t = core.SparseArray()
    u = core.SparseArray()
    t &= u
    assert s is t
    s = t = core.SparseArray()
    u = core.SparseArray()
    t |= u
    assert s is t
    s = t = core.SparseArray()
    u = core.SparseArray()
    t ^= u
    assert s is t

def test_sparse_array_nonzero():
    if False:
        while True:
            i = 10
    sa = core.SparseArray()
    assert not sa
    sa.set_bit(0)
    assert sa
    sa = core.SparseArray.all_on()
    assert sa
    sa.clear_range(0, 100)
    assert sa

def test_sparse_array_getstate():
    if False:
        i = 10
        return i + 15
    sa = core.SparseArray()
    assert sa.__getstate__() == ()
    sa = core.SparseArray()
    sa.invert_in_place()
    assert sa.__getstate__() == (0,)
    sa = core.SparseArray()
    sa.set_range(0, 2)
    sa.set_range(4, 4)
    assert sa.__getstate__() == (0, 2, 4, 8)
    sa = core.SparseArray()
    sa.invert_in_place()
    sa.clear_range(2, 4)
    assert sa.__getstate__() == (0, 2, 6)
    sa = core.SparseArray()
    sa.invert_in_place()
    sa.clear_range(0, 2)
    sa.clear_range(4, 4)
    assert sa.__getstate__() == (2, 4, 8)

def test_sparse_array_pickle():
    if False:
        for i in range(10):
            print('nop')
    sa = core.SparseArray()
    assert sa == pickle.loads(pickle.dumps(sa, -1))
    sa = core.SparseArray()
    sa.invert_in_place()
    assert sa == pickle.loads(pickle.dumps(sa, -1))
    sa = core.SparseArray()
    sa.set_range(0, 2)
    sa.set_range(4, 4)
    assert sa == pickle.loads(pickle.dumps(sa, -1))
    sa = core.SparseArray()
    sa.invert_in_place()
    sa.clear_range(2, 4)
    assert sa == pickle.loads(pickle.dumps(sa, -1))
    sa = core.SparseArray()
    sa.invert_in_place()
    sa.clear_range(0, 2)
    sa.clear_range(4, 4)
    assert sa == pickle.loads(pickle.dumps(sa, -1))