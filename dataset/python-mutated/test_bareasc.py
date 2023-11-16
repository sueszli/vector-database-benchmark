"""Tests for the 'bareasc' plugin."""
import unittest
from test.helper import TestHelper, capture_stdout
from beets import logging

class BareascPluginTest(unittest.TestCase, TestHelper):
    """Test bare ASCII query matching."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Set up test environment for bare ASCII query matching.'
        self.setup_beets()
        self.log = logging.getLogger('beets.web')
        self.config['bareasc']['prefix'] = '#'
        self.load_plugins('bareasc')
        self.add_item(title='with accents', album_id=2, artist='Antonín Dvořák')
        self.add_item(title='without accents', artist='Antonín Dvorak')
        self.add_item(title='with umlaut', album_id=2, artist='Brüggen')
        self.add_item(title='without umlaut or e', artist='Bruggen')
        self.add_item(title='without umlaut with e', artist='Brueggen')

    def test_search_normal_noaccent(self):
        if False:
            return 10
        'Normal search, no accents, not using bare-ASCII match.\n\n        Finds just the unaccented entry.\n        '
        items = self.lib.items('dvorak')
        self.assertEqual(len(items), 1)
        self.assertEqual([items[0].title], ['without accents'])

    def test_search_normal_accent(self):
        if False:
            return 10
        'Normal search, with accents, not using bare-ASCII match.\n\n        Finds just the accented entry.\n        '
        items = self.lib.items('dvořák')
        self.assertEqual(len(items), 1)
        self.assertEqual([items[0].title], ['with accents'])

    def test_search_bareasc_noaccent(self):
        if False:
            return 10
        'Bare-ASCII search, no accents.\n\n        Finds both entries.\n        '
        items = self.lib.items('#dvorak')
        self.assertEqual(len(items), 2)
        self.assertEqual({items[0].title, items[1].title}, {'without accents', 'with accents'})

    def test_search_bareasc_accent(self):
        if False:
            i = 10
            return i + 15
        'Bare-ASCII search, with accents.\n\n        Finds both entries.\n        '
        items = self.lib.items('#dvořák')
        self.assertEqual(len(items), 2)
        self.assertEqual({items[0].title, items[1].title}, {'without accents', 'with accents'})

    def test_search_bareasc_wrong_accent(self):
        if False:
            i = 10
            return i + 15
        'Bare-ASCII search, with incorrect accent.\n\n        Finds both entries.\n        '
        items = self.lib.items('#dvořäk')
        self.assertEqual(len(items), 2)
        self.assertEqual({items[0].title, items[1].title}, {'without accents', 'with accents'})

    def test_search_bareasc_noumlaut(self):
        if False:
            i = 10
            return i + 15
        "Bare-ASCII search, with no umlaut.\n\n        Finds entry with 'u' not 'ue', although German speaker would\n        normally replace ü with ue.\n\n        This is expected behaviour for this simple plugin.\n        "
        items = self.lib.items('#Bruggen')
        self.assertEqual(len(items), 2)
        self.assertEqual({items[0].title, items[1].title}, {'without umlaut or e', 'with umlaut'})

    def test_search_bareasc_umlaut(self):
        if False:
            while True:
                i = 10
        "Bare-ASCII search, with umlaut.\n\n        Finds entry with 'u' not 'ue', although German speaker would\n        normally replace ü with ue.\n\n        This is expected behaviour for this simple plugin.\n        "
        items = self.lib.items('#Brüggen')
        self.assertEqual(len(items), 2)
        self.assertEqual({items[0].title, items[1].title}, {'without umlaut or e', 'with umlaut'})

    def test_bareasc_list_output(self):
        if False:
            print('Hello World!')
        'Bare-ASCII version of list command - check output.'
        with capture_stdout() as output:
            self.run_command('bareasc', 'with accents')
        self.assertIn('Antonin Dvorak', output.getvalue())

    def test_bareasc_format_output(self):
        if False:
            while True:
                i = 10
        'Bare-ASCII version of list -f command - check output.'
        with capture_stdout() as output:
            self.run_command('bareasc', 'with accents', '-f', '$artist:: $title')
        self.assertEqual('Antonin Dvorak:: with accents\n', output.getvalue())

def suite():
    if False:
        while True:
            i = 10
    'loader.'
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')