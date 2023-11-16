from __future__ import absolute_import
from oslo_config import cfg
import os
import unittest2
import yaml
from yaml import SafeLoader, FullLoader
try:
    from yaml import CSafeLoader
except ImportError:
    CSafeLoader = None
from mock import Mock
from st2common.content.loader import ContentPackLoader
from st2common.content.loader import OverrideLoader
from st2common.content.loader import LOG
from st2common.constants.meta import yaml_safe_load
from st2tests import config
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
RESOURCES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../resources'))

class ContentLoaderTest(unittest2.TestCase):

    def test_get_sensors(self):
        if False:
            while True:
                i = 10
        packs_base_path = os.path.join(RESOURCES_DIR, 'packs/')
        loader = ContentPackLoader()
        pack_sensors = loader.get_content(base_dirs=[packs_base_path], content_type='sensors')
        self.assertIsNotNone(pack_sensors.get('pack1', None))

    def test_get_sensors_pack_missing_sensors(self):
        if False:
            print('Hello World!')
        loader = ContentPackLoader()
        fail_pack_path = os.path.join(RESOURCES_DIR, 'packs/pack2')
        self.assertTrue(os.path.exists(fail_pack_path))
        self.assertEqual(loader._get_sensors(fail_pack_path), None)

    def test_invalid_content_type(self):
        if False:
            for i in range(10):
                print('nop')
        packs_base_path = os.path.join(RESOURCES_DIR, 'packs/')
        loader = ContentPackLoader()
        self.assertRaises(ValueError, loader.get_content, base_dirs=[packs_base_path], content_type='stuff')

    def test_get_content_multiple_directories(self):
        if False:
            while True:
                i = 10
        packs_base_path_1 = os.path.join(RESOURCES_DIR, 'packs/')
        packs_base_path_2 = os.path.join(RESOURCES_DIR, 'packs2/')
        base_dirs = [packs_base_path_1, packs_base_path_2]
        LOG.warning = Mock()
        loader = ContentPackLoader()
        sensors = loader.get_content(base_dirs=base_dirs, content_type='sensors')
        self.assertIn('pack1', sensors)
        self.assertIn('pack3', sensors)
        expected_msg = 'Pack "pack1" already found in "%s/packs/", ignoring content from "%s/packs2/"' % (RESOURCES_DIR, RESOURCES_DIR)
        LOG.warning.assert_called_once_with(expected_msg)

    def test_get_content_from_pack_success(self):
        if False:
            for i in range(10):
                print('nop')
        loader = ContentPackLoader()
        pack_path = os.path.join(RESOURCES_DIR, 'packs/pack1')
        sensors = loader.get_content_from_pack(pack_dir=pack_path, content_type='sensors')
        self.assertTrue(sensors.endswith('packs/pack1/sensors'))

    def test_get_content_from_pack_directory_doesnt_exist(self):
        if False:
            print('Hello World!')
        loader = ContentPackLoader()
        pack_path = os.path.join(RESOURCES_DIR, 'packs/pack100')
        message_regex = "Directory .*? doesn't exist"
        self.assertRaisesRegexp(ValueError, message_regex, loader.get_content_from_pack, pack_dir=pack_path, content_type='sensors')

    def test_get_content_from_pack_no_sensors(self):
        if False:
            return 10
        loader = ContentPackLoader()
        pack_path = os.path.join(RESOURCES_DIR, 'packs/pack2')
        result = loader.get_content_from_pack(pack_dir=pack_path, content_type='sensors')
        self.assertEqual(result, None)

    def test_get_override_action_from_default(self):
        if False:
            i = 10
            return i + 15
        config.parse_args()
        cfg.CONF.set_override(name='base_path', override=RESOURCES_DIR, group='system')
        loader = OverrideLoader()
        content = {'name': 'action1', 'enabled': True}
        self.assertTrue(loader.override('overpack1', 'actions', content))
        self.assertFalse(content['enabled'])
        content = {'name': 'action1', 'enabled': False}
        self.assertFalse(loader.override('overpack1', 'actions', content))
        self.assertFalse(content['enabled'])

    def test_get_override_action_from_exception(self):
        if False:
            while True:
                i = 10
        config.parse_args()
        cfg.CONF.set_override(name='base_path', override=RESOURCES_DIR, group='system')
        loader = OverrideLoader()
        content = {'name': 'action2', 'enabled': True}
        self.assertFalse(loader.override('overpack1', 'actions', content))
        self.assertTrue(content['enabled'])
        content = {'name': 'action2', 'enabled': False}
        self.assertTrue(loader.override('overpack1', 'actions', content))
        self.assertTrue(content['enabled'])

    def test_get_override_action_from_default_no_exceptions(self):
        if False:
            i = 10
            return i + 15
        config.parse_args()
        cfg.CONF.set_override(name='base_path', override=RESOURCES_DIR, group='system')
        loader = OverrideLoader()
        content = {'name': 'action1', 'enabled': True}
        self.assertTrue(loader.override('overpack4', 'actions', content))
        self.assertFalse(content['enabled'])
        content = {'name': 'action2', 'enabled': True}
        self.assertTrue(loader.override('overpack4', 'actions', content))
        self.assertFalse(content['enabled'])

    def test_get_override_action_from_global_default_no_exceptions(self):
        if False:
            while True:
                i = 10
        config.parse_args()
        cfg.CONF.set_override(name='base_path', override=RESOURCES_DIR, group='system')
        loader = OverrideLoader()
        content = {'class_name': 'sensor1', 'enabled': True}
        self.assertTrue(loader.override('overpack1', 'sensors', content))
        self.assertFalse(content['enabled'])

    def test_get_override_action_from_global_overridden_by_pack(self):
        if False:
            print('Hello World!')
        config.parse_args()
        cfg.CONF.set_override(name='base_path', override=RESOURCES_DIR, group='system')
        loader = OverrideLoader()
        content = {'class_name': 'sensor1', 'enabled': True}
        self.assertFalse(loader.override('overpack2', 'sensors', content))
        self.assertTrue(content['enabled'])

    def test_get_override_action_from_global_overridden_by_pack_exception(self):
        if False:
            while True:
                i = 10
        config.parse_args()
        cfg.CONF.set_override(name='base_path', override=RESOURCES_DIR, group='system')
        loader = OverrideLoader()
        content = {'class_name': 'sensor1', 'enabled': True}
        self.assertFalse(loader.override('overpack3', 'sensors', content))
        self.assertTrue(content['enabled'])

    def test_get_override_invalid_type(self):
        if False:
            print('Hello World!')
        config.parse_args()
        cfg.CONF.set_override(name='base_path', override=RESOURCES_DIR, group='system')
        loader = OverrideLoader()
        content = {'name': 'action2', 'enabled': True}
        self.assertRaises(ValueError, loader.override, pack_name='overpack1', resource_type='wrongtype', content=content)

    def test_get_override_invalid_default_key(self):
        if False:
            print('Hello World!')
        config.parse_args()
        cfg.CONF.set_override(name='base_path', override=RESOURCES_DIR, group='system')
        loader = OverrideLoader()
        content = {'name': 'action1', 'enabled': True}
        self.assertRaises(ValueError, loader.override, pack_name='overpack2', resource_type='actions', content=content)

    def test_get_override_invalid_exceptions_key(self):
        if False:
            while True:
                i = 10
        config.parse_args()
        cfg.CONF.set_override(name='base_path', override=RESOURCES_DIR, group='system')
        loader = OverrideLoader()
        content = {'name': 'action1', 'enabled': True}
        loader.override('overpack1', 'actions', content)
        content = {'name': 'action2', 'enabled': True}
        self.assertRaises(ValueError, loader.override, pack_name='overpack3', resource_type='actions', content=content)

class YamlLoaderTestCase(unittest2.TestCase):

    def test_yaml_safe_load(self):
        if False:
            return 10
        dumped = yaml.dump(Foo)
        self.assertTrue('!!python' in dumped)
        result = yaml.load(dumped, Loader=FullLoader)
        self.assertTrue(result)
        self.assertRaisesRegexp(yaml.constructor.ConstructorError, 'could not determine a constructor', yaml_safe_load, dumped)
        self.assertRaisesRegexp(yaml.constructor.ConstructorError, 'could not determine a constructor', yaml.load, dumped, Loader=SafeLoader)
        if CSafeLoader:
            self.assertRaisesRegexp(yaml.constructor.ConstructorError, 'could not determine a constructor', yaml.load, dumped, Loader=CSafeLoader)

class Foo(object):
    a = '1'
    b = 'c'