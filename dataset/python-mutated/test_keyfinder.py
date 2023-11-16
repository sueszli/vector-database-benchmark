import unittest
from test.helper import TestHelper
from unittest.mock import patch
from beets import util
from beets.library import Item

@patch('beets.util.command_output')
class KeyFinderTest(unittest.TestCase, TestHelper):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_beets()
        self.load_plugins('keyfinder')

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.teardown_beets()
        self.unload_plugins()

    def test_add_key(self, command_output):
        if False:
            i = 10
            return i + 15
        item = Item(path='/file')
        item.add(self.lib)
        command_output.return_value = util.CommandOutput(b'dbm', b'')
        self.run_command('keyfinder')
        item.load()
        self.assertEqual(item['initial_key'], 'C#m')
        command_output.assert_called_with(['KeyFinder', '-f', util.syspath(item.path)])

    def test_add_key_on_import(self, command_output):
        if False:
            i = 10
            return i + 15
        command_output.return_value = util.CommandOutput(b'dbm', b'')
        importer = self.create_importer()
        importer.run()
        item = self.lib.items().get()
        self.assertEqual(item['initial_key'], 'C#m')

    def test_force_overwrite(self, command_output):
        if False:
            i = 10
            return i + 15
        self.config['keyfinder']['overwrite'] = True
        item = Item(path='/file', initial_key='F')
        item.add(self.lib)
        command_output.return_value = util.CommandOutput(b'C#m', b'')
        self.run_command('keyfinder')
        item.load()
        self.assertEqual(item['initial_key'], 'C#m')

    def test_do_not_overwrite(self, command_output):
        if False:
            return 10
        item = Item(path='/file', initial_key='F')
        item.add(self.lib)
        command_output.return_value = util.CommandOutput(b'dbm', b'')
        self.run_command('keyfinder')
        item.load()
        self.assertEqual(item['initial_key'], 'F')

    def test_no_key(self, command_output):
        if False:
            return 10
        item = Item(path='/file')
        item.add(self.lib)
        command_output.return_value = util.CommandOutput(b'', b'')
        self.run_command('keyfinder')
        item.load()
        self.assertEqual(item['initial_key'], None)

def suite():
    if False:
        while True:
            i = 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')