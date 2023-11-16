"""Test module for file ui/__init__.py
"""
import os
import shutil
import unittest
from copy import deepcopy
from random import random
from test import _common
from test.helper import control_stdin
from beets import config, ui

class InputMethodsTest(_common.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.io.install()

    def _print_helper(self, s):
        if False:
            i = 10
            return i + 15
        print(s)

    def _print_helper2(self, s, prefix):
        if False:
            for i in range(10):
                print('nop')
        print(prefix, s)

    def test_input_select_objects(self):
        if False:
            i = 10
            return i + 15
        full_items = ['1', '2', '3', '4', '5']
        self.io.addinput('n')
        items = ui.input_select_objects('Prompt', full_items, self._print_helper)
        self.assertEqual(items, [])
        self.io.addinput('y')
        items = ui.input_select_objects('Prompt', full_items, self._print_helper)
        self.assertEqual(items, full_items)
        self.io.addinput('s')
        self.io.addinput('n')
        self.io.addinput('y')
        self.io.addinput('n')
        self.io.addinput('y')
        self.io.addinput('n')
        items = ui.input_select_objects('Prompt', full_items, self._print_helper)
        self.assertEqual(items, ['2', '4'])
        self.io.addinput('s')
        self.io.addinput('y')
        self.io.addinput('y')
        self.io.addinput('n')
        self.io.addinput('y')
        self.io.addinput('n')
        items = ui.input_select_objects('Prompt', full_items, lambda s: self._print_helper2(s, 'Prefix'))
        self.assertEqual(items, ['1', '2', '4'])
        self.io.addinput('s')
        self.io.addinput('y')
        self.io.addinput('n')
        self.io.addinput('y')
        self.io.addinput('q')
        items = ui.input_select_objects('Prompt', full_items, self._print_helper)
        self.assertEqual(items, ['1', '3'])

class InitTest(_common.LibTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()

    def test_human_bytes(self):
        if False:
            return 10
        tests = [(0, '0.0 B'), (30, '30.0 B'), (pow(2, 10), '1.0 KiB'), (pow(2, 20), '1.0 MiB'), (pow(2, 30), '1.0 GiB'), (pow(2, 40), '1.0 TiB'), (pow(2, 50), '1.0 PiB'), (pow(2, 60), '1.0 EiB'), (pow(2, 70), '1.0 ZiB'), (pow(2, 80), '1.0 YiB'), (pow(2, 90), '1.0 HiB'), (pow(2, 100), 'big')]
        for (i, h) in tests:
            self.assertEqual(h, ui.human_bytes(i))

    def test_human_seconds(self):
        if False:
            while True:
                i = 10
        tests = [(0, '0.0 seconds'), (30, '30.0 seconds'), (60, '1.0 minutes'), (90, '1.5 minutes'), (125, '2.1 minutes'), (3600, '1.0 hours'), (86400, '1.0 days'), (604800, '1.0 weeks'), (31449600, '1.0 years'), (314496000, '1.0 decades')]
        for (i, h) in tests:
            self.assertEqual(h, ui.human_seconds(i))

class ParentalDirCreation(_common.TestCase):

    def test_create_yes(self):
        if False:
            i = 10
            return i + 15
        non_exist_path = _common.util.py3_path(os.path.join(self.temp_dir, b'nonexist', str(random()).encode()))
        test_config = deepcopy(config)
        test_config['library'] = non_exist_path
        with control_stdin('y'):
            lib = ui._open_library(test_config)
        lib._close()

    def test_create_no(self):
        if False:
            print('Hello World!')
        non_exist_path_parent = _common.util.py3_path(os.path.join(self.temp_dir, b'nonexist'))
        non_exist_path = _common.util.py3_path(os.path.join(non_exist_path_parent.encode(), str(random()).encode()))
        test_config = deepcopy(config)
        test_config['library'] = non_exist_path
        with control_stdin('n'):
            try:
                lib = ui._open_library(test_config)
            except ui.UserError:
                if os.path.exists(non_exist_path_parent):
                    shutil.rmtree(non_exist_path_parent)
                    raise OSError('Parent directories should not be created.')
            else:
                if lib:
                    lib._close()
                raise OSError('Parent directories should not be created.')

def suite():
    if False:
        print('Hello World!')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')