"""Tests for the 'acousticbrainz' plugin.
"""
import json
import os.path
import unittest
from test._common import RSRC
from beetsplug.acousticbrainz import ABSCHEME, AcousticPlugin

class MapDataToSchemeTest(unittest.TestCase):

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        ab = AcousticPlugin()
        data = {'key 1': 'value 1', 'key 2': 'value 2'}
        scheme = {'key 1': 'attribute 1', 'key 2': 'attribute 2'}
        mapping = set(ab._map_data_to_scheme(data, scheme))
        self.assertEqual(mapping, {('attribute 1', 'value 1'), ('attribute 2', 'value 2')})

    def test_recurse(self):
        if False:
            print('Hello World!')
        ab = AcousticPlugin()
        data = {'key': 'value', 'group': {'subkey': 'subvalue', 'subgroup': {'subsubkey': 'subsubvalue'}}}
        scheme = {'key': 'attribute 1', 'group': {'subkey': 'attribute 2', 'subgroup': {'subsubkey': 'attribute 3'}}}
        mapping = set(ab._map_data_to_scheme(data, scheme))
        self.assertEqual(mapping, {('attribute 1', 'value'), ('attribute 2', 'subvalue'), ('attribute 3', 'subsubvalue')})

    def test_composite(self):
        if False:
            return 10
        ab = AcousticPlugin()
        data = {'key 1': 'part 1', 'key 2': 'part 2'}
        scheme = {'key 1': ('attribute', 0), 'key 2': ('attribute', 1)}
        mapping = set(ab._map_data_to_scheme(data, scheme))
        self.assertEqual(mapping, {('attribute', 'part 1 part 2')})

    def test_realistic(self):
        if False:
            for i in range(10):
                print('nop')
        ab = AcousticPlugin()
        data_path = os.path.join(RSRC, b'acousticbrainz/data.json')
        with open(data_path) as res:
            data = json.load(res)
        mapping = set(ab._map_data_to_scheme(data, ABSCHEME))
        expected = {('chords_key', 'A'), ('average_loudness', 0.815025985241), ('mood_acoustic', 0.415711194277), ('chords_changes_rate', 0.0445116683841), ('tonal', 0.874250173569), ('mood_sad', 0.299694597721), ('bpm', 162.532119751), ('gender', 'female'), ('initial_key', 'A minor'), ('chords_number_rate', 0.00194468453992), ('mood_relaxed', 0.123632438481), ('chords_scale', 'minor'), ('voice_instrumental', 'instrumental'), ('key_strength', 0.636936545372), ('genre_rosamerica', 'roc'), ('mood_party', 0.234383180737), ('mood_aggressive', 0.0779221653938), ('danceable', 0.143928021193), ('rhythm', 'VienneseWaltz'), ('mood_electronic', 0.339881360531), ('mood_happy', 0.0894767045975), ('moods_mirex', 'Cluster3'), ('timbre', 'bright')}
        self.assertEqual(mapping, expected)

def suite():
    if False:
        while True:
            i = 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')