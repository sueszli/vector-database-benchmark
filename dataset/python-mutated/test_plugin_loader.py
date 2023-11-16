from __future__ import absolute_import
import abc
import copy
import os
import six
import sys
import unittest2
import st2common.util.loader as plugin_loader
PLUGIN_FOLDER = 'loadableplugin'
SRC_RELATIVE = os.path.join('../resources', PLUGIN_FOLDER)
SRC_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), SRC_RELATIVE)

class LoaderTest(unittest2.TestCase):
    sys_path = None

    @six.add_metaclass(abc.ABCMeta)
    class DummyPlugin(object):
        """
        Base class that test plugins should implement
        """

        @abc.abstractmethod
        def do_work(self):
            if False:
                i = 10
                return i + 15
            pass

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        LoaderTest.sys_path = copy.copy(sys.path)

    def tearDown(self):
        if False:
            return 10
        sys.path = LoaderTest.sys_path

    def test_module_load_from_file(self):
        if False:
            for i in range(10):
                print('nop')
        plugin_path = os.path.join(SRC_ROOT, 'plugin/standaloneplugin.py')
        plugin_classes = plugin_loader.register_plugin(LoaderTest.DummyPlugin, plugin_path)
        self.assertEqual(1, len(plugin_classes))
        self.assertIn(os.path.abspath(os.path.join(SRC_ROOT, 'plugin')), sys.path)
        for plugin_class in plugin_classes:
            try:
                plugin_instance = plugin_class()
                ret_val = plugin_instance.do_work()
                self.assertIsNotNone(ret_val, 'Should be non-null.')
            except:
                pass

    def test_module_load_from_file_fail(self):
        if False:
            i = 10
            return i + 15
        try:
            plugin_path = os.path.join(SRC_ROOT, 'plugin/sampleplugin.py')
            plugin_loader.register_plugin(LoaderTest.DummyPlugin, plugin_path)
            self.assertTrue(False, 'Import error is expected.')
        except ImportError:
            self.assertTrue(True)

    def test_syspath_unchanged_load_multiple_plugins(self):
        if False:
            i = 10
            return i + 15
        plugin_1_path = os.path.join(SRC_ROOT, 'plugin/sampleplugin.py')
        try:
            plugin_loader.register_plugin(LoaderTest.DummyPlugin, plugin_1_path)
        except ImportError:
            pass
        old_sys_path = copy.copy(sys.path)
        plugin_2_path = os.path.join(SRC_ROOT, 'plugin/sampleplugin2.py')
        try:
            plugin_loader.register_plugin(LoaderTest.DummyPlugin, plugin_2_path)
        except ImportError:
            pass
        self.assertEqual(old_sys_path, sys.path, 'Should be equal.')

    def test_register_plugin_class_class_doesnt_exist(self):
        if False:
            print('Hello World!')
        file_path = os.path.join(SRC_ROOT, 'plugin/sampleplugin3.py')
        expected_msg = 'doesn\'t expose class named "SamplePluginNotExists"'
        self.assertRaisesRegexp(Exception, expected_msg, plugin_loader.register_plugin_class, base_class=LoaderTest.DummyPlugin, file_path=file_path, class_name='SamplePluginNotExists')

    def test_register_plugin_class_abstract_method_not_implemented(self):
        if False:
            while True:
                i = 10
        file_path = os.path.join(SRC_ROOT, 'plugin/sampleplugin3.py')
        expected_msg = 'doesn\'t implement required "do_work" method from the base class'
        self.assertRaisesRegexp(plugin_loader.IncompatiblePluginException, expected_msg, plugin_loader.register_plugin_class, base_class=LoaderTest.DummyPlugin, file_path=file_path, class_name='SamplePlugin')