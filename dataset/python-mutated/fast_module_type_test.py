"""Tests for tensorflow.python.util.fast_module_type."""
from tensorflow.python.platform import test
from tensorflow.python.util import fast_module_type
FastModuleType = fast_module_type.get_fast_module_type_class()

class ChildFastModule(FastModuleType):

    def _getattribute1(self, name):
        if False:
            print('Hello World!')
        return 2

    def _getattribute2(self, name):
        if False:
            i = 10
            return i + 15
        raise AttributeError('Pass to getattr')

    def _getattr(self, name):
        if False:
            return 10
        return 3

class FastModuleTypeTest(test.TestCase):

    def testBaseGetattribute(self):
        if False:
            for i in range(10):
                print('nop')
        module = ChildFastModule('test')
        module.foo = 1
        self.assertEqual(1, module.foo)

    def testGetattributeCallback(self):
        if False:
            i = 10
            return i + 15
        module = ChildFastModule('test')
        FastModuleType.set_getattribute_callback(module, ChildFastModule._getattribute1)
        self.assertEqual(2, module.foo)

    def testGetattrCallback(self):
        if False:
            return 10
        module = ChildFastModule('test')
        FastModuleType.set_getattribute_callback(module, ChildFastModule._getattribute2)
        FastModuleType.set_getattr_callback(module, ChildFastModule._getattr)
        self.assertEqual(3, module.foo)

    def testFastdictApis(self):
        if False:
            while True:
                i = 10
        module = ChildFastModule('test')
        self.assertFalse(module._fastdict_key_in('bar'))
        with self.assertRaisesRegex(KeyError, "module has no attribute 'bar'"):
            module._fastdict_get('bar')
        module._fastdict_insert('bar', 1)
        self.assertTrue(module._fastdict_key_in('bar'))
        self.assertEqual(1, module.bar)
if __name__ == '__main__':
    test.main()