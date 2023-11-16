"""
Testing C implementation of the numba typed-list
"""
import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
LIST_OK = 0
LIST_ERR_INDEX = -1
LIST_ERR_NO_MEMORY = -2
LIST_ERR_MUTATED = -3
LIST_ERR_ITER_EXHAUSTED = -4
LIST_ERR_IMMUTABLE = -5

class List(object):
    """A wrapper around the C-API to provide a minimal list object for
    testing.
    """

    def __init__(self, tc, item_size, allocated):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        tc : TestCase instance\n        item_size : int\n            byte size for the items\n        allocated : int\n            number of items to allocate for\n        '
        self.tc = tc
        self.item_size = item_size
        self.lp = self.list_new(item_size, allocated)

    def __del__(self):
        if False:
            print('Hello World!')
        self.tc.numba_list_free(self.lp)

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.list_length()

    def __setitem__(self, i, item):
        if False:
            for i in range(10):
                print('nop')
        return self.list_setitem(i, item)

    def __getitem__(self, i):
        if False:
            return 10
        return self.list_getitem(i)

    def __iter__(self):
        if False:
            print('Hello World!')
        return ListIter(self)

    def __delitem__(self, i):
        if False:
            i = 10
            return i + 15
        self.list_delitem(i)

    def handle_index(self, i):
        if False:
            for i in range(10):
                print('nop')
        if i < -1 or len(self) == 0:
            IndexError('list index out of range')
        elif i == -1:
            i = len(self) - 1
        return i

    @property
    def allocated(self):
        if False:
            print('Hello World!')
        return self.list_allocated()

    @property
    def is_mutable(self):
        if False:
            i = 10
            return i + 15
        return self.list_is_mutable()

    def set_mutable(self):
        if False:
            for i in range(10):
                print('nop')
        return self.list_set_is_mutable(1)

    def set_immutable(self):
        if False:
            i = 10
            return i + 15
        return self.list_set_is_mutable(0)

    def append(self, item):
        if False:
            while True:
                i = 10
        self.list_append(item)

    def pop(self, i=-1):
        if False:
            while True:
                i = 10
        return self.list_pop(i)

    def list_new(self, item_size, allocated):
        if False:
            for i in range(10):
                print('nop')
        lp = ctypes.c_void_p()
        status = self.tc.numba_list_new(ctypes.byref(lp), item_size, allocated)
        self.tc.assertEqual(status, LIST_OK)
        return lp

    def list_length(self):
        if False:
            print('Hello World!')
        return self.tc.numba_list_length(self.lp)

    def list_allocated(self):
        if False:
            return 10
        return self.tc.numba_list_allocated(self.lp)

    def list_is_mutable(self):
        if False:
            print('Hello World!')
        return self.tc.numba_list_is_mutable(self.lp)

    def list_set_is_mutable(self, is_mutable):
        if False:
            print('Hello World!')
        return self.tc.numba_list_set_is_mutable(self.lp, is_mutable)

    def list_setitem(self, i, item):
        if False:
            print('Hello World!')
        status = self.tc.numba_list_setitem(self.lp, i, item)
        if status == LIST_ERR_INDEX:
            raise IndexError('list index out of range')
        elif status == LIST_ERR_IMMUTABLE:
            raise ValueError('list is immutable')
        else:
            self.tc.assertEqual(status, LIST_OK)

    def list_getitem(self, i):
        if False:
            return 10
        i = self.handle_index(i)
        item_out_buffer = ctypes.create_string_buffer(self.item_size)
        status = self.tc.numba_list_getitem(self.lp, i, item_out_buffer)
        if status == LIST_ERR_INDEX:
            raise IndexError('list index out of range')
        else:
            self.tc.assertEqual(status, LIST_OK)
            return item_out_buffer.raw

    def list_append(self, item):
        if False:
            return 10
        status = self.tc.numba_list_append(self.lp, item)
        if status == LIST_ERR_IMMUTABLE:
            raise ValueError('list is immutable')
        self.tc.assertEqual(status, LIST_OK)

    def list_pop(self, i):
        if False:
            i = 10
            return i + 15
        i = self.handle_index(i)
        item = self.list_getitem(i)
        self.list_delitem(i)
        return item

    def list_delitem(self, i):
        if False:
            while True:
                i = 10
        if isinstance(i, slice):
            status = self.tc.numba_list_delete_slice(self.lp, i.start, i.stop, i.step)
            if status == LIST_ERR_IMMUTABLE:
                raise ValueError('list is immutable')
            self.tc.assertEqual(status, LIST_OK)
        else:
            i = self.handle_index(i)
            status = self.tc.numba_list_delitem(self.lp, i)
            if status == LIST_ERR_INDEX:
                raise IndexError('list index out of range')
            elif status == LIST_ERR_IMMUTABLE:
                raise ValueError('list is immutable')
            self.tc.assertEqual(status, LIST_OK)

    def list_iter(self, itptr):
        if False:
            i = 10
            return i + 15
        self.tc.numba_list_iter(itptr, self.lp)

    def list_iter_next(self, itptr):
        if False:
            i = 10
            return i + 15
        bi = ctypes.c_void_p(0)
        status = self.tc.numba_list_iter_next(itptr, ctypes.byref(bi))
        if status == LIST_ERR_MUTATED:
            raise ValueError('list mutated')
        elif status == LIST_ERR_ITER_EXHAUSTED:
            raise StopIteration
        else:
            self.tc.assertGreaterEqual(status, 0)
            item = (ctypes.c_char * self.item_size).from_address(bi.value)
            return item.value

class ListIter(object):
    """An iterator for the `List`.
    """

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        self.parent = parent
        itsize = self.parent.tc.numba_list_iter_sizeof()
        self.it_state_buf = (ctypes.c_char_p * itsize)(0)
        self.it = ctypes.cast(self.it_state_buf, ctypes.c_void_p)
        self.parent.list_iter(self.it)

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        return self.parent.list_iter_next(self.it)
    next = __next__

class TestListImpl(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        'Bind to the c_helper library and provide the ctypes wrapper.\n        '
        list_t = ctypes.c_void_p
        iter_t = ctypes.c_void_p

        def wrap(name, restype, argtypes=()):
            if False:
                for i in range(10):
                    print('nop')
            proto = ctypes.CFUNCTYPE(restype, *argtypes)
            return proto(_helperlib.c_helpers[name])
        self.numba_test_list = wrap('test_list', ctypes.c_int)
        self.numba_list_new = wrap('list_new', ctypes.c_int, [ctypes.POINTER(list_t), ctypes.c_ssize_t, ctypes.c_ssize_t])
        self.numba_list_free = wrap('list_free', None, [list_t])
        self.numba_list_length = wrap('list_length', ctypes.c_int, [list_t])
        self.numba_list_allocated = wrap('list_allocated', ctypes.c_int, [list_t])
        self.numba_list_is_mutable = wrap('list_is_mutable', ctypes.c_int, [list_t])
        self.numba_list_set_is_mutable = wrap('list_set_is_mutable', None, [list_t, ctypes.c_int])
        self.numba_list_setitem = wrap('list_setitem', ctypes.c_int, [list_t, ctypes.c_ssize_t, ctypes.c_char_p])
        self.numba_list_append = wrap('list_append', ctypes.c_int, [list_t, ctypes.c_char_p])
        self.numba_list_getitem = wrap('list_getitem', ctypes.c_int, [list_t, ctypes.c_ssize_t, ctypes.c_char_p])
        self.numba_list_delitem = wrap('list_delitem', ctypes.c_int, [list_t, ctypes.c_ssize_t])
        self.numba_list_delete_slice = wrap('list_delete_slice', ctypes.c_int, [list_t, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t])
        self.numba_list_iter_sizeof = wrap('list_iter_sizeof', ctypes.c_size_t)
        self.numba_list_iter = wrap('list_iter', None, [iter_t, list_t])
        self.numba_list_iter_next = wrap('list_iter_next', ctypes.c_int, [iter_t, ctypes.POINTER(ctypes.c_void_p)])

    def test_simple_c_test(self):
        if False:
            for i in range(10):
                print('nop')
        ret = self.numba_test_list()
        self.assertEqual(ret, 0)

    def test_length(self):
        if False:
            return 10
        l = List(self, 8, 0)
        self.assertEqual(len(l), 0)

    def test_allocation(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(16):
            l = List(self, 8, i)
            self.assertEqual(len(l), 0)
            self.assertEqual(l.allocated, i)

    def test_append_get_string(self):
        if False:
            return 10
        l = List(self, 8, 1)
        l.append(b'abcdefgh')
        self.assertEqual(len(l), 1)
        r = l[0]
        self.assertEqual(r, b'abcdefgh')

    def test_append_get_int(self):
        if False:
            print('Hello World!')
        l = List(self, 8, 1)
        l.append(struct.pack('q', 1))
        self.assertEqual(len(l), 1)
        r = struct.unpack('q', l[0])[0]
        self.assertEqual(r, 1)

    def test_append_get_string_realloc(self):
        if False:
            for i in range(10):
                print('nop')
        l = List(self, 8, 1)
        l.append(b'abcdefgh')
        self.assertEqual(len(l), 1)
        l.append(b'hijklmno')
        self.assertEqual(len(l), 2)
        r = l[1]
        self.assertEqual(r, b'hijklmno')

    def test_set_item_getitem_index_error(self):
        if False:
            return 10
        l = List(self, 8, 0)
        with self.assertRaises(IndexError):
            l[0]
        with self.assertRaises(IndexError):
            l[0] = b'abcdefgh'

    def test_iter(self):
        if False:
            return 10
        l = List(self, 1, 0)
        values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']
        for i in values:
            l.append(i)
        received = []
        for j in l:
            received.append(j)
        self.assertEqual(values, received)

    def test_pop(self):
        if False:
            for i in range(10):
                print('nop')
        l = List(self, 1, 0)
        values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']
        for i in values:
            l.append(i)
        self.assertEqual(len(l), 8)
        received = l.pop()
        self.assertEqual(b'h', received)
        self.assertEqual(len(l), 7)
        received = [j for j in l]
        self.assertEqual(received, values[:-1])
        received = l.pop(0)
        self.assertEqual(b'a', received)
        self.assertEqual(len(l), 6)
        received = l.pop(2)
        self.assertEqual(b'd', received)
        self.assertEqual(len(l), 5)
        expected = [b'b', b'c', b'e', b'f', b'g']
        received = [j for j in l]
        self.assertEqual(received, expected)

    def test_pop_index_error(self):
        if False:
            print('Hello World!')
        l = List(self, 8, 0)
        with self.assertRaises(IndexError):
            l.pop()

    def test_pop_byte(self):
        if False:
            print('Hello World!')
        l = List(self, 4, 0)
        values = [b'aaaa', b'bbbb', b'cccc', b'dddd', b'eeee', b'ffff', b'gggg', b'hhhhh']
        for i in values:
            l.append(i)
        self.assertEqual(len(l), 8)
        received = l.pop()
        self.assertEqual(b'hhhh', received)
        self.assertEqual(len(l), 7)
        received = [j for j in l]
        self.assertEqual(received, values[:-1])
        received = l.pop(0)
        self.assertEqual(b'aaaa', received)
        self.assertEqual(len(l), 6)
        received = l.pop(2)
        self.assertEqual(b'dddd', received)
        self.assertEqual(len(l), 5)
        expected = [b'bbbb', b'cccc', b'eeee', b'ffff', b'gggg']
        received = [j for j in l]
        self.assertEqual(received, expected)

    def test_delitem(self):
        if False:
            print('Hello World!')
        l = List(self, 1, 0)
        values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']
        for i in values:
            l.append(i)
        self.assertEqual(len(l), 8)
        del l[0]
        self.assertEqual(len(l), 7)
        self.assertEqual(list(l), values[1:])
        del l[-1]
        self.assertEqual(len(l), 6)
        self.assertEqual(list(l), values[1:-1])
        del l[2]
        self.assertEqual(len(l), 5)
        self.assertEqual(list(l), [b'b', b'c', b'e', b'f', b'g'])

    def test_delete_slice(self):
        if False:
            print('Hello World!')
        l = List(self, 1, 0)
        values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']
        for i in values:
            l.append(i)
        self.assertEqual(len(l), 8)
        del l[0:8:2]
        self.assertEqual(len(l), 4)
        self.assertEqual(list(l), values[1:8:2])
        del l[0:1:1]
        self.assertEqual(len(l), 3)
        self.assertEqual(list(l), [b'd', b'f', b'h'])
        del l[2:3:1]
        self.assertEqual(len(l), 2)
        self.assertEqual(list(l), [b'd', b'f'])
        del l[0:2:1]
        self.assertEqual(len(l), 0)
        self.assertEqual(list(l), [])

    def check_sizing(self, item_size, nmax):
        if False:
            print('Hello World!')
        l = List(self, item_size, 0)

        def make_item(v):
            if False:
                for i in range(10):
                    print('nop')
            tmp = '{:0{}}'.format(nmax - v - 1, item_size).encode('latin-1')
            return tmp[:item_size]
        for i in range(nmax):
            l.append(make_item(i))
        self.assertEqual(len(l), nmax)
        for i in range(nmax):
            self.assertEqual(l[i], make_item(i))

    def test_sizing(self):
        if False:
            return 10
        for i in range(1, 16):
            self.check_sizing(item_size=i, nmax=2 ** i)

    def test_mutability(self):
        if False:
            while True:
                i = 10
        l = List(self, 8, 1)
        one = struct.pack('q', 1)
        l.append(one)
        self.assertTrue(l.is_mutable)
        self.assertEqual(len(l), 1)
        r = struct.unpack('q', l[0])[0]
        self.assertEqual(r, 1)
        l.set_immutable()
        self.assertFalse(l.is_mutable)
        with self.assertRaises(ValueError) as raises:
            l.append(one)
        self.assertIn('list is immutable', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            l[0] = one
        self.assertIn('list is immutable', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            l.pop()
        self.assertIn('list is immutable', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            del l[0]
        self.assertIn('list is immutable', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            del l[0:1:1]
        self.assertIn('list is immutable', str(raises.exception))
        l.set_mutable()
        self.assertTrue(l.is_mutable)
        self.assertEqual(len(l), 1)
        r = struct.unpack('q', l[0])[0]
        self.assertEqual(r, 1)