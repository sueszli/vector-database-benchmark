from runner.koan import *

class AboutDictionaries(Koan):

    def test_creating_dictionaries(self):
        if False:
            for i in range(10):
                print('nop')
        empty_dict = dict()
        self.assertEqual(dict, type(empty_dict))
        self.assertDictEqual({}, empty_dict)
        self.assertEqual(__, len(empty_dict))

    def test_dictionary_literals(self):
        if False:
            print('Hello World!')
        empty_dict = {}
        self.assertEqual(dict, type(empty_dict))
        babel_fish = {'one': 'uno', 'two': 'dos'}
        self.assertEqual(__, len(babel_fish))

    def test_accessing_dictionaries(self):
        if False:
            print('Hello World!')
        babel_fish = {'one': 'uno', 'two': 'dos'}
        self.assertEqual(__, babel_fish['one'])
        self.assertEqual(__, babel_fish['two'])

    def test_changing_dictionaries(self):
        if False:
            i = 10
            return i + 15
        babel_fish = {'one': 'uno', 'two': 'dos'}
        babel_fish['one'] = 'eins'
        expected = {'two': 'dos', 'one': __}
        self.assertDictEqual(expected, babel_fish)

    def test_dictionary_is_unordered(self):
        if False:
            print('Hello World!')
        dict1 = {'one': 'uno', 'two': 'dos'}
        dict2 = {'two': 'dos', 'one': 'uno'}
        self.assertEqual(__, dict1 == dict2)

    def test_dictionary_keys_and_values(self):
        if False:
            i = 10
            return i + 15
        babel_fish = {'one': 'uno', 'two': 'dos'}
        self.assertEqual(__, len(babel_fish.keys()))
        self.assertEqual(__, len(babel_fish.values()))
        self.assertEqual(__, 'one' in babel_fish.keys())
        self.assertEqual(__, 'two' in babel_fish.values())
        self.assertEqual(__, 'uno' in babel_fish.keys())
        self.assertEqual(__, 'dos' in babel_fish.values())

    def test_making_a_dictionary_from_a_sequence_of_keys(self):
        if False:
            for i in range(10):
                print('nop')
        cards = {}.fromkeys(('red warrior', 'green elf', 'blue valkyrie', 'yellow dwarf', 'confused looking zebra'), 42)
        self.assertEqual(__, len(cards))
        self.assertEqual(__, cards['green elf'])
        self.assertEqual(__, cards['yellow dwarf'])