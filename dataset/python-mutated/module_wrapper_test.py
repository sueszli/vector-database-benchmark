"""Tests for tensorflow.python.util.module_wrapper."""
import pickle
import types
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import module_wrapper
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2
module_wrapper._PER_MODULE_WARNING_LIMIT = 5

class MockModule(types.ModuleType):
    pass

class DeprecationWrapperTest(test.TestCase):

    def testWrapperIsAModule(self):
        if False:
            while True:
                i = 10
        module = MockModule('test')
        wrapped_module = module_wrapper.TFModuleWrapper(module, 'test')
        self.assertTrue(tf_inspect.ismodule(wrapped_module))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def testDeprecationWarnings(self, mock_warning):
        if False:
            return 10
        module = MockModule('test')
        module.foo = 1
        module.bar = 2
        module.baz = 3
        all_renames_v2.symbol_renames['tf.test.bar'] = 'tf.bar2'
        all_renames_v2.symbol_renames['tf.test.baz'] = 'tf.compat.v1.baz'
        wrapped_module = module_wrapper.TFModuleWrapper(module, 'test')
        self.assertTrue(tf_inspect.ismodule(wrapped_module))
        self.assertEqual(0, mock_warning.call_count)
        bar = wrapped_module.bar
        self.assertEqual(1, mock_warning.call_count)
        foo = wrapped_module.foo
        self.assertEqual(1, mock_warning.call_count)
        baz = wrapped_module.baz
        self.assertEqual(2, mock_warning.call_count)
        baz = wrapped_module.baz
        self.assertEqual(2, mock_warning.call_count)
        self.assertEqual(module.foo, foo)
        self.assertEqual(module.bar, bar)

class LazyLoadingWrapperTest(test.TestCase):

    def testLazyLoad(self):
        if False:
            print('Hello World!')
        module = MockModule('test')
        apis = {'cmd': ('', 'cmd'), 'ABCMeta': ('abc', 'ABCMeta')}
        wrapped_module = module_wrapper.TFModuleWrapper(module, 'test', public_apis=apis, deprecation=False)
        import cmd as _cmd
        from abc import ABCMeta as _ABCMeta
        self.assertFalse(wrapped_module._fastdict_key_in('cmd'))
        self.assertEqual(wrapped_module.cmd, _cmd)
        self.assertTrue(wrapped_module._fastdict_key_in('cmd'))
        self.assertFalse(wrapped_module._fastdict_key_in('ABCMeta'))
        self.assertEqual(wrapped_module.ABCMeta, _ABCMeta)
        self.assertTrue(wrapped_module._fastdict_key_in('ABCMeta'))

    def testLazyLoadLocalOverride(self):
        if False:
            print('Hello World!')
        module = MockModule('test')
        apis = {'cmd': ('', 'cmd')}
        wrapped_module = module_wrapper.TFModuleWrapper(module, 'test', public_apis=apis, deprecation=False)
        import cmd as _cmd
        self.assertEqual(wrapped_module.cmd, _cmd)
        setattr(wrapped_module, 'cmd', 1)
        setattr(wrapped_module, 'cgi', 2)
        self.assertEqual(wrapped_module.cmd, 1)
        self.assertEqual(wrapped_module._fastdict_get('cmd'), 1)
        self.assertEqual(wrapped_module.cgi, 2)
        self.assertEqual(wrapped_module._fastdict_get('cgi'), 2)

    def testLazyLoadDict(self):
        if False:
            return 10
        module = MockModule('test')
        apis = {'cmd': ('', 'cmd')}
        wrapped_module = module_wrapper.TFModuleWrapper(module, 'test', public_apis=apis, deprecation=False)
        import cmd as _cmd
        self.assertNotIn('cmd', wrapped_module.__dict__)
        wrapped_module.cmd
        self.assertEqual(wrapped_module.__dict__['cmd'], _cmd)
        setattr(wrapped_module, 'cmd2', _cmd)
        self.assertEqual(wrapped_module.__dict__['cmd2'], _cmd)

    def testLazyLoadWildcardImport(self):
        if False:
            for i in range(10):
                print('nop')
        module = MockModule('test')
        module._should_not_be_public = 5
        apis = {'cmd': ('', 'cmd')}
        wrapped_module = module_wrapper.TFModuleWrapper(module, 'test', public_apis=apis, deprecation=False)
        setattr(wrapped_module, 'hello', 1)
        self.assertIn('hello', wrapped_module.__all__)
        self.assertIn('cmd', wrapped_module.__all__)
        self.assertNotIn('_should_not_be_public', wrapped_module.__all__)

    def testLazyLoadCorrectLiteModule(self):
        if False:
            return 10
        module = MockModule('test')
        apis = {'lite': ('', 'cmd')}
        module.lite = 5
        import cmd as _cmd
        wrapped_module = module_wrapper.TFModuleWrapper(module, 'test', public_apis=apis, deprecation=False, has_lite=True)
        self.assertEqual(wrapped_module.lite, _cmd)

    def testInitCachesAttributes(self):
        if False:
            i = 10
            return i + 15
        module = MockModule('test')
        wrapped_module = module_wrapper.TFModuleWrapper(module, 'test')
        self.assertTrue(wrapped_module._fastdict_key_in('_fastdict_key_in'))
        self.assertTrue(wrapped_module._fastdict_key_in('_tfmw_module_name'))
        self.assertTrue(wrapped_module._fastdict_key_in('__all__'))

    def testCompatV1APIInstrumenting(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)
        apis = {'cosh': ('', 'cmd')}
        mock_tf = MockModule('tensorflow')
        mock_tf_wrapped = module_wrapper.TFModuleWrapper(mock_tf, 'test', public_apis=apis)
        mock_tf_wrapped.cosh
        self.assertFalse(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)
        mock_tf_v1 = MockModule('tensorflow.compat.v1')
        mock_tf_v1_wrapped = module_wrapper.TFModuleWrapper(mock_tf_v1, 'test', public_apis=apis)
        self.assertFalse(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)
        mock_tf_v1_wrapped.cosh
        self.assertTrue(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)
        module_wrapper.TFModuleWrapper.compat_v1_usage_recorded = False
        mock_tf_v2_v1 = mock_tf_v1 = MockModule('tensorflow.compat.v2.compat.v1')
        mock_tf_v2_v1_wrapped = module_wrapper.TFModuleWrapper(mock_tf_v2_v1, 'test', public_apis=apis)
        self.assertFalse(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)
        mock_tf_v2_v1_wrapped.cosh
        self.assertTrue(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)

    def testDelAttr(self):
        if False:
            return 10
        module = MockModule('test')
        wrapped_module = module_wrapper.TFModuleWrapper(module, 'test')
        setattr(wrapped_module, 'foo', 1)
        self.assertEqual(wrapped_module.foo, 1)
        delattr(wrapped_module, 'foo')
        self.assertFalse(hasattr(wrapped_module, 'foo'))
        setattr(wrapped_module, 'foo', 1)
        self.assertEqual(wrapped_module.foo, 1)
        delattr(wrapped_module, 'foo')
        self.assertFalse(hasattr(wrapped_module, 'foo'))

class PickleTest(test.TestCase):

    def testPickleSubmodule(self):
        if False:
            print('Hello World!')
        name = PickleTest.__module__
        module = module_wrapper.TFModuleWrapper(MockModule(name), name)
        restored = pickle.loads(pickle.dumps(module))
        self.assertEqual(restored.__name__, name)
        self.assertIsNotNone(restored.PickleTest)
if __name__ == '__main__':
    test.main()