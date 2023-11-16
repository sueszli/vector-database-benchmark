"""Unit tests for Workflow.settings API."""
from __future__ import print_function, unicode_literals, absolute_import
import json
import os
import shutil
import time
import tempfile
import unittest
from workflow.workflow import Settings
from tests.util import DEFAULT_SETTINGS

class SettingsTests(unittest.TestCase):
    """Test suite for `workflow.workflow.Settings`."""

    def setUp(self):
        if False:
            return 10
        'Initialise unit test environment.'
        self.tempdir = tempfile.mkdtemp()
        self.settings_file = os.path.join(self.tempdir, 'settings.json')
        with open(self.settings_file, 'wb') as file_obj:
            json.dump(DEFAULT_SETTINGS, file_obj)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        'Reset unit test environment.'
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    def test_defaults(self):
        if False:
            for i in range(10):
                print('nop')
        'Default settings'
        if os.path.exists(self.settings_file):
            os.unlink(self.settings_file)
        s = Settings(self.settings_file, {'key1': 'value2'})
        self.assertEqual(s['key1'], 'value2')

    def test_load_settings(self):
        if False:
            while True:
                i = 10
        'Load saved settings'
        s = Settings(self.settings_file, {'key1': 'value2'})
        for key in DEFAULT_SETTINGS:
            self.assertEqual(DEFAULT_SETTINGS[key], s[key])

    def test_save_settings(self):
        if False:
            print('Hello World!')
        'Settings saved'
        s = Settings(self.settings_file)
        self.assertEqual(s['key1'], DEFAULT_SETTINGS['key1'])
        s['key1'] = 'spoons!'
        s2 = Settings(self.settings_file)
        self.assertEqual(s['key1'], s2['key1'])

    def test_delete_settings(self):
        if False:
            return 10
        'Settings deleted'
        s = Settings(self.settings_file)
        self.assertEqual(s['key1'], DEFAULT_SETTINGS['key1'])
        del s['key1']
        s2 = Settings(self.settings_file)
        self.assertEqual(s2.get('key1'), None)

    def test_dict_methods(self):
        if False:
            print('Hello World!')
        'Settings dict methods'
        other = {'key1': 'spoons!'}
        s = Settings(self.settings_file)
        self.assertEqual(s['key1'], DEFAULT_SETTINGS['key1'])
        s.update(other)
        s.setdefault('alist', [])
        s2 = Settings(self.settings_file)
        self.assertEqual(s['key1'], s2['key1'])
        self.assertEqual(s['key1'], 'spoons!')
        self.assertEqual(s2['alist'], [])

    def test_settings_not_rewritten(self):
        if False:
            print('Hello World!')
        'Settings not rewritten for same value'
        s = Settings(self.settings_file)
        mt = os.path.getmtime(self.settings_file)
        time.sleep(1)
        now = time.time()
        for (k, v) in DEFAULT_SETTINGS.items():
            s[k] = v
        self.assertTrue(os.path.getmtime(self.settings_file) == mt)
        s['finished_at'] = now
        s2 = Settings(self.settings_file)
        self.assertEqual(s['finished_at'], s2['finished_at'])
        self.assertTrue(os.path.getmtime(self.settings_file) > mt)

    def test_mutable_objects_updated(self):
        if False:
            return 10
        'Updated mutable objects cause save'
        s = Settings(self.settings_file)
        mt1 = os.path.getmtime(self.settings_file)
        time.sleep(1)
        seq = s['mutable1']
        seq.append('another string')
        s['mutable1'] = seq
        mt2 = os.path.getmtime(self.settings_file)
        self.assertTrue(mt2 > mt1)
        s2 = Settings(self.settings_file)
        self.assertTrue('another string' in s2['mutable1'])
if __name__ == '__main__':
    unittest.main()