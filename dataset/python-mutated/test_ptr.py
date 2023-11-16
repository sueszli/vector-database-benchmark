import ctypes
from unittest import mock
from django.contrib.gis.ptr import CPointerBase
from django.test import SimpleTestCase

class CPointerBaseTests(SimpleTestCase):

    def test(self):
        if False:
            i = 10
            return i + 15
        destructor_mock = mock.Mock()

        class NullPointerException(Exception):
            pass

        class FakeGeom1(CPointerBase):
            null_ptr_exception_class = NullPointerException

        class FakeGeom2(FakeGeom1):
            ptr_type = ctypes.POINTER(ctypes.c_float)
            destructor = destructor_mock
        fg1 = FakeGeom1()
        fg2 = FakeGeom2()
        fg1.ptr = fg1.ptr_type()
        fg1.ptr = None
        fg2.ptr = fg2.ptr_type(ctypes.c_float(5.23))
        fg2.ptr = None
        for fg in (fg1, fg2):
            with self.assertRaises(NullPointerException):
                fg.ptr
        bad_ptrs = (5, ctypes.c_char_p(b'foobar'))
        for bad_ptr in bad_ptrs:
            for fg in (fg1, fg2):
                with self.assertRaisesMessage(TypeError, 'Incompatible pointer type'):
                    fg.ptr = bad_ptr
        fg = FakeGeom1()
        fg.ptr = fg.ptr_type(1)
        del fg
        fg = FakeGeom2()
        fg.ptr = None
        del fg
        self.assertFalse(destructor_mock.called)
        fg = FakeGeom2()
        ptr = fg.ptr_type(ctypes.c_float(1.0))
        fg.ptr = ptr
        del fg
        destructor_mock.assert_called_with(ptr)

    def test_destructor_catches_importerror(self):
        if False:
            i = 10
            return i + 15

        class FakeGeom(CPointerBase):
            destructor = mock.Mock(side_effect=ImportError)
        fg = FakeGeom()
        fg.ptr = fg.ptr_type(1)
        del fg