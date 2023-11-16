"""Tests for c_api utils."""
import gc
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

class ApiDefMapTest(test_util.TensorFlowTestCase):

    def testApiDefMapOpNames(self):
        if False:
            print('Hello World!')
        api_def_map = c_api_util.ApiDefMap()
        self.assertIn('Add', api_def_map.op_names())

    def testApiDefMapGet(self):
        if False:
            return 10
        api_def_map = c_api_util.ApiDefMap()
        op_def = api_def_map.get_op_def('Add')
        self.assertEqual(op_def.name, 'Add')
        api_def = api_def_map.get_api_def('Add')
        self.assertEqual(api_def.graph_op_name, 'Add')

    def testApiDefMapPutThenGet(self):
        if False:
            i = 10
            return i + 15
        api_def_map = c_api_util.ApiDefMap()
        api_def_text = '\nop {\n  graph_op_name: "Add"\n  summary: "Returns x + y element-wise."\n  description: <<END\n*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting\n[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)\nEND\n}\n'
        api_def_map.put_api_def(api_def_text)
        api_def = api_def_map.get_api_def('Add')
        self.assertEqual(api_def.graph_op_name, 'Add')
        self.assertEqual(api_def.summary, 'Returns x + y element-wise.')

class UniquePtrTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            return 10
        super(UniquePtrTest, self).setUp()

        class MockClass:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.deleted = False

        def deleter(obj):
            if False:
                for i in range(10):
                    print('nop')
            obj.deleted = True
        self.obj = MockClass()
        self.deleter = deleter

    def testLifeCycle(self):
        if False:
            print('Hello World!')
        self.assertFalse(self.obj.deleted)
        a = c_api_util.UniquePtr(name='mock', deleter=self.deleter, obj=self.obj)
        with a.get() as obj:
            self.assertIs(obj, self.obj)
        del a
        gc.collect()
        self.assertTrue(self.obj.deleted)

    def testSafeUnderRaceCondition(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.obj.deleted)
        a = c_api_util.UniquePtr(name='mock', deleter=self.deleter, obj=self.obj)
        with a.get() as obj:
            self.assertIs(obj, self.obj)
            del a
            gc.collect()
            self.assertFalse(obj.deleted)
        gc.collect()
        self.assertTrue(self.obj.deleted)

    def testRaiseAfterDeleted(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(self.obj.deleted)
        a = c_api_util.UniquePtr(name='mock', deleter=self.deleter, obj=self.obj)
        a.__del__()
        self.assertTrue(self.obj.deleted)
        with self.assertRaisesRegex(c_api_util.AlreadyGarbageCollectedError, 'MockClass'):
            with a.get():
                pass
        gc.collect()
        self.assertTrue(self.obj.deleted)
if __name__ == '__main__':
    googletest.main()