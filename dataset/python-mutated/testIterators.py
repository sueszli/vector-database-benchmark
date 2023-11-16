import sys
import unittest
import pythoncom
import win32com.server.util
import win32com.test.util
from win32com.client import Dispatch
from win32com.client.gencache import EnsureDispatch

class _BaseTestCase(win32com.test.util.TestCase):

    def test_enumvariant_vb(self):
        if False:
            while True:
                i = 10
        (ob, iter) = self.iter_factory()
        got = []
        for v in iter:
            got.append(v)
        self.assertEqual(got, self.expected_data)

    def test_yield(self):
        if False:
            i = 10
            return i + 15
        (ob, i) = self.iter_factory()
        got = []
        for v in iter(i):
            got.append(v)
        self.assertEqual(got, self.expected_data)

    def _do_test_nonenum(self, object):
        if False:
            print('Hello World!')
        try:
            for i in object:
                pass
            self.fail('Could iterate over a non-iterable object')
        except TypeError:
            pass
        self.assertRaises(TypeError, iter, object)
        self.assertRaises(AttributeError, getattr, object, 'next')

    def test_nonenum_wrapper(self):
        if False:
            while True:
                i = 10
        ob = self.object._oleobj_
        try:
            for i in ob:
                pass
            self.fail('Could iterate over a non-iterable object')
        except TypeError:
            pass
        self.assertRaises(TypeError, iter, ob)
        self.assertRaises(AttributeError, getattr, ob, 'next')
        ob = self.object
        try:
            for i in ob:
                pass
            self.fail('Could iterate over a non-iterable object')
        except TypeError:
            pass
        try:
            next(iter(ob))
            self.fail('Expected a TypeError fetching this iterator')
        except TypeError:
            pass
        self.assertRaises(AttributeError, getattr, ob, 'next')

class VBTestCase(_BaseTestCase):

    def setUp(self):
        if False:
            print('Hello World!')

        def factory():
            if False:
                print('Hello World!')
            ob = self.object.EnumerableCollectionProperty
            for i in self.expected_data:
                ob.Add(i)
            invkind = pythoncom.DISPATCH_METHOD | pythoncom.DISPATCH_PROPERTYGET
            iter = ob._oleobj_.InvokeTypes(pythoncom.DISPID_NEWENUM, 0, invkind, (13, 10), ())
            return (ob, iter.QueryInterface(pythoncom.IID_IEnumVARIANT))
        self.object = EnsureDispatch('PyCOMVBTest.Tester')
        self.expected_data = [1, 'Two', '3']
        self.iter_factory = factory

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.object = None

class SomeObject:
    _public_methods_ = ['GetCollection']

    def __init__(self, data):
        if False:
            return 10
        self.data = data

    def GetCollection(self):
        if False:
            print('Hello World!')
        return win32com.server.util.NewCollection(self.data)

class WrappedPythonCOMServerTestCase(_BaseTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')

        def factory():
            if False:
                i = 10
                return i + 15
            ob = self.object.GetCollection()
            flags = pythoncom.DISPATCH_METHOD | pythoncom.DISPATCH_PROPERTYGET
            enum = ob._oleobj_.Invoke(pythoncom.DISPID_NEWENUM, 0, flags, 1)
            return (ob, enum.QueryInterface(pythoncom.IID_IEnumVARIANT))
        self.expected_data = [1, 'Two', 3]
        sv = win32com.server.util.wrap(SomeObject(self.expected_data))
        self.object = Dispatch(sv)
        self.iter_factory = factory

    def tearDown(self):
        if False:
            return 10
        self.object = None

def suite():
    if False:
        return 10
    suite = unittest.TestSuite()
    for item in list(globals().values()):
        if isinstance(item, type) and issubclass(item, unittest.TestCase) and (item != _BaseTestCase):
            suite.addTest(unittest.makeSuite(item))
    return suite
if __name__ == '__main__':
    unittest.main(argv=sys.argv + ['suite'])