"""Tests for naming module."""
import unittest
from nvidia.dali._autograph.pyct import naming

class NamerTest(unittest.TestCase):

    def test_new_symbol_tracks_names(self):
        if False:
            i = 10
            return i + 15
        namer = naming.Namer({})
        self.assertEqual('temp', namer.new_symbol('temp', set()))
        self.assertEqual(('temp',), tuple(sorted(namer.generated_names)))

    def test_new_symbol_avoids_duplicates(self):
        if False:
            return 10
        namer = naming.Namer({})
        self.assertEqual('temp', namer.new_symbol('temp', set()))
        self.assertEqual('temp_1', namer.new_symbol('temp', set()))
        self.assertEqual(('temp', 'temp_1'), tuple(sorted(namer.generated_names)))

    def test_new_symbol_avoids_conflicts(self):
        if False:
            return 10
        namer = naming.Namer({'temp': 1})
        self.assertEqual('temp_1', namer.new_symbol('temp', set()))
        self.assertEqual('temp_3', namer.new_symbol('temp', set(('temp_2',))))
        self.assertEqual(('temp_1', 'temp_3'), tuple(sorted(namer.generated_names)))