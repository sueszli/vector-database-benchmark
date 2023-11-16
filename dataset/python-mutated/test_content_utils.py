from __future__ import absolute_import
import os
import os.path
import unittest2
from oslo_config import cfg
from st2common.constants.action import LIBS_DIR as ACTION_LIBS_DIR
from st2common.content.utils import get_pack_base_path
from st2common.content.utils import get_packs_base_paths
from st2common.content.utils import get_aliases_base_paths
from st2common.content.utils import get_pack_resource_file_abs_path
from st2common.content.utils import get_pack_file_abs_path
from st2common.content.utils import get_entry_point_abs_path
from st2common.content.utils import get_action_libs_abs_path
from st2common.content.utils import get_relative_path_to_pack_file
from st2tests import config as tests_config
from st2tests.fixturesloader import get_fixtures_packs_base_path
from st2tests.fixtures.packs.dummy_pack_1.fixture import PACK_NAME as DUMMY_PACK_1, PACK_PATH as DUMMY_PACK_1_PATH
from st2tests.fixtures.packs.dummy_pack_2.fixture import PACK_PATH as DUMMY_PACK_2_PATH

class ContentUtilsTestCase(unittest2.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        tests_config.parse_args()

    def test_get_pack_base_paths(self):
        if False:
            print('Hello World!')
        cfg.CONF.content.system_packs_base_path = ''
        cfg.CONF.content.packs_base_paths = '/opt/path1'
        result = get_packs_base_paths()
        self.assertEqual(result, ['/opt/path1'])
        cfg.CONF.content.packs_base_paths = '/opt/path1:/opt/path2'
        result = get_packs_base_paths()
        self.assertEqual(result, ['/opt/path1', '/opt/path2'])
        cfg.CONF.content.packs_base_paths = '/opt/path1:/opt/path2:'
        result = get_packs_base_paths()
        self.assertEqual(result, ['/opt/path1', '/opt/path2'])
        cfg.CONF.content.packs_base_paths = '/opt/path1:/opt/path2:/opt/path1:/opt/path2'
        result = get_packs_base_paths()
        self.assertEqual(result, ['/opt/path1', '/opt/path2'])
        cfg.CONF.content.system_packs_base_path = '/opt/system'
        cfg.CONF.content.packs_base_paths = '/opt/path2:/opt/path1'
        result = get_packs_base_paths()
        self.assertEqual(result, ['/opt/system', '/opt/path2', '/opt/path1'])
        orig_path = cfg.CONF.content.system_packs_base_path
        cfg.CONF.content.system_packs_base_path = '/tests/packs'
        names = ['test_pack_1', 'test_pack_2', 'ma_pack']
        for name in names:
            actual = get_pack_base_path(pack_name=name)
            expected = os.path.join(cfg.CONF.content.system_packs_base_path, name)
            self.assertEqual(actual, expected)
        cfg.CONF.content.system_packs_base_path = orig_path

    def test_get_aliases_base_paths(self):
        if False:
            for i in range(10):
                print('nop')
        cfg.CONF.content.aliases_base_paths = '/opt/path1'
        result = get_aliases_base_paths()
        self.assertEqual(result, ['/opt/path1'])
        cfg.CONF.content.aliases_base_paths = '/opt/path1:/opt/path2'
        result = get_aliases_base_paths()
        self.assertEqual(result, ['/opt/path1', '/opt/path2'])
        cfg.CONF.content.aliases_base_paths = '/opt/path1:/opt/path2:'
        result = get_aliases_base_paths()
        self.assertEqual(result, ['/opt/path1', '/opt/path2'])
        cfg.CONF.content.aliases_base_paths = '/opt/path1:/opt/path2:/opt/path1:/opt/path2'
        result = get_aliases_base_paths()
        self.assertEqual(result, ['/opt/path1', '/opt/path2'])

    def test_get_pack_resource_file_abs_path(self):
        if False:
            for i in range(10):
                print('nop')
        cfg.CONF.content.packs_base_paths = get_fixtures_packs_base_path()
        expected_msg = 'Invalid resource type: fooo'
        self.assertRaisesRegexp(ValueError, expected_msg, get_pack_resource_file_abs_path, pack_ref=DUMMY_PACK_1, resource_type='fooo', file_path='test.py')
        file_paths = ['/tmp/foo.py', '../foo.py', '/etc/passwd', '../../foo.py', '/opt/stackstorm/packs/invalid_pack/actions/my_action.py', '../../foo.py']
        for file_path in file_paths:
            expected_msg = 'Invalid file path: ".*%s"\\. File path needs to be relative to the pack actions directory (.*). For example "my_action.py"\\.' % file_path
            self.assertRaisesRegexp(ValueError, expected_msg, get_pack_resource_file_abs_path, pack_ref=DUMMY_PACK_1, resource_type='action', file_path=file_path)
            expected_msg = 'Invalid file path: ".*%s"\\. File path needs to be relative to the pack sensors directory (.*). For example "my_sensor.py"\\.' % file_path
            self.assertRaisesRegexp(ValueError, expected_msg, get_pack_resource_file_abs_path, pack_ref=DUMMY_PACK_1, resource_type='sensor', file_path=file_path)
            expected_msg = 'Invalid file path: ".*%s"\\. File path needs to be relative to the pack directory (.*). For example "my_action.py"\\.' % file_path
            self.assertRaisesRegexp(ValueError, expected_msg, get_pack_file_abs_path, pack_ref=DUMMY_PACK_1, file_path=file_path)
        file_paths = ['foo.py', 'a/foo.py', 'a/b/foo.py']
        for file_path in file_paths:
            expected = os.path.join(DUMMY_PACK_1_PATH, 'actions', file_path)
            result = get_pack_resource_file_abs_path(pack_ref=DUMMY_PACK_1, resource_type='action', file_path=file_path)
            self.assertEqual(result, expected)

    def test_get_entry_point_absolute_path(self):
        if False:
            while True:
                i = 10
        orig_path = cfg.CONF.content.system_packs_base_path
        cfg.CONF.content.system_packs_base_path = '/tests/packs'
        acutal_path = get_entry_point_abs_path(pack='foo', entry_point='/tests/packs/foo/bar.py')
        self.assertEqual(acutal_path, '/tests/packs/foo/bar.py', "Entry point path doesn't match.")
        cfg.CONF.content.system_packs_base_path = orig_path

    def test_get_entry_point_absolute_path_empty(self):
        if False:
            i = 10
            return i + 15
        orig_path = cfg.CONF.content.system_packs_base_path
        cfg.CONF.content.system_packs_base_path = '/tests/packs'
        acutal_path = get_entry_point_abs_path(pack='foo', entry_point=None)
        self.assertEqual(acutal_path, None, "Entry point path doesn't match.")
        acutal_path = get_entry_point_abs_path(pack='foo', entry_point='')
        self.assertEqual(acutal_path, None, "Entry point path doesn't match.")
        cfg.CONF.content.system_packs_base_path = orig_path

    def test_get_entry_point_relative_path(self):
        if False:
            return 10
        orig_path = cfg.CONF.content.system_packs_base_path
        cfg.CONF.content.system_packs_base_path = '/tests/packs'
        acutal_path = get_entry_point_abs_path(pack='foo', entry_point='foo/bar.py')
        expected_path = os.path.join(cfg.CONF.content.system_packs_base_path, 'foo', 'actions', 'foo/bar.py')
        self.assertEqual(acutal_path, expected_path, "Entry point path doesn't match.")
        cfg.CONF.content.system_packs_base_path = orig_path

    def test_get_action_libs_abs_path(self):
        if False:
            while True:
                i = 10
        orig_path = cfg.CONF.content.system_packs_base_path
        cfg.CONF.content.system_packs_base_path = '/tests/packs'
        acutal_path = get_action_libs_abs_path(pack='foo', entry_point='foo/bar.py')
        expected_path = os.path.join(cfg.CONF.content.system_packs_base_path, 'foo', 'actions', os.path.join('foo', ACTION_LIBS_DIR))
        self.assertEqual(acutal_path, expected_path, "Action libs path doesn't match.")
        acutal_path = get_action_libs_abs_path(pack='foo', entry_point='/tests/packs/foo/tmp/foo.py')
        expected_path = os.path.join('/tests/packs/foo/tmp', ACTION_LIBS_DIR)
        self.assertEqual(acutal_path, expected_path, "Action libs path doesn't match.")
        cfg.CONF.content.system_packs_base_path = orig_path

    def test_get_relative_path_to_pack_file(self):
        if False:
            for i in range(10):
                print('nop')
        packs_base_paths = get_fixtures_packs_base_path()
        pack_ref = DUMMY_PACK_1
        file_path = os.path.join(DUMMY_PACK_1_PATH, 'pack.yaml')
        result = get_relative_path_to_pack_file(pack_ref=pack_ref, file_path=file_path)
        self.assertEqual(result, 'pack.yaml')
        file_path = os.path.join(DUMMY_PACK_1_PATH, 'actions/action.meta.yaml')
        result = get_relative_path_to_pack_file(pack_ref=pack_ref, file_path=file_path)
        self.assertEqual(result, 'actions/action.meta.yaml')
        file_path = os.path.join(DUMMY_PACK_1_PATH, 'actions/lib/foo.py')
        result = get_relative_path_to_pack_file(pack_ref=pack_ref, file_path=file_path)
        self.assertEqual(result, 'actions/lib/foo.py')
        file_path = 'actions/lib/foo2.py'
        result = get_relative_path_to_pack_file(pack_ref=pack_ref, file_path=file_path)
        self.assertEqual(result, 'actions/lib/foo2.py')
        expected_msg = 'file_path (.*?) is not located inside the pack directory (.*?)'
        file_path = os.path.join(DUMMY_PACK_2_PATH, 'actions/lib/foo.py')
        self.assertRaisesRegexp(ValueError, expected_msg, get_relative_path_to_pack_file, pack_ref=pack_ref, file_path=file_path)
        file_path = '/tmp/foo/bar.py'
        self.assertRaisesRegexp(ValueError, expected_msg, get_relative_path_to_pack_file, pack_ref=pack_ref, file_path=file_path)
        file_path = os.path.join(packs_base_paths, '../dummy_pack_1/pack.yaml')
        self.assertRaisesRegexp(ValueError, expected_msg, get_relative_path_to_pack_file, pack_ref=pack_ref, file_path=file_path)
        file_path = os.path.join(packs_base_paths, '../../dummy_pack_1/pack.yaml')
        self.assertRaisesRegexp(ValueError, expected_msg, get_relative_path_to_pack_file, pack_ref=pack_ref, file_path=file_path)