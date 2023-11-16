"""Tests for naming module."""
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.platform import test

class NamerTest(test.TestCase):

    def test_new_symbol_tracks_names(self):
        if False:
            i = 10
            return i + 15
        namer = naming.Namer({})
        self.assertEqual('temp', namer.new_symbol('temp', set()))
        self.assertItemsEqual(('temp',), namer.generated_names)

    def test_new_symbol_avoids_duplicates(self):
        if False:
            return 10
        namer = naming.Namer({})
        self.assertEqual('temp', namer.new_symbol('temp', set()))
        self.assertEqual('temp_1', namer.new_symbol('temp', set()))
        self.assertItemsEqual(('temp', 'temp_1'), namer.generated_names)

    def test_new_symbol_avoids_conflicts(self):
        if False:
            return 10
        namer = naming.Namer({'temp': 1})
        self.assertEqual('temp_1', namer.new_symbol('temp', set()))
        self.assertEqual('temp_3', namer.new_symbol('temp', set(('temp_2',))))
        self.assertItemsEqual(('temp_1', 'temp_3'), namer.generated_names)
if __name__ == '__main__':
    test.main()