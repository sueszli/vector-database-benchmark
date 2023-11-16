from django.template.defaultfilters import dictsortreversed
from django.test import SimpleTestCase

class FunctionTests(SimpleTestCase):

    def test_sort(self):
        if False:
            for i in range(10):
                print('nop')
        sorted_dicts = dictsortreversed([{'age': 23, 'name': 'Barbara-Ann'}, {'age': 63, 'name': 'Ra Ra Rasputin'}, {'name': 'Jonny B Goode', 'age': 18}], 'age')
        self.assertEqual([sorted(dict.items()) for dict in sorted_dicts], [[('age', 63), ('name', 'Ra Ra Rasputin')], [('age', 23), ('name', 'Barbara-Ann')], [('age', 18), ('name', 'Jonny B Goode')]])

    def test_sort_list_of_tuples(self):
        if False:
            return 10
        data = [('a', '42'), ('c', 'string'), ('b', 'foo')]
        expected = [('c', 'string'), ('b', 'foo'), ('a', '42')]
        self.assertEqual(dictsortreversed(data, 0), expected)

    def test_sort_list_of_tuple_like_dicts(self):
        if False:
            return 10
        data = [{'0': 'a', '1': '42'}, {'0': 'c', '1': 'string'}, {'0': 'b', '1': 'foo'}]
        expected = [{'0': 'c', '1': 'string'}, {'0': 'b', '1': 'foo'}, {'0': 'a', '1': '42'}]
        self.assertEqual(dictsortreversed(data, '0'), expected)

    def test_invalid_values(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If dictsortreversed is passed something other than a list of\n        dictionaries, fail silently.\n        '
        self.assertEqual(dictsortreversed([1, 2, 3], 'age'), '')
        self.assertEqual(dictsortreversed('Hello!', 'age'), '')
        self.assertEqual(dictsortreversed({'a': 1}, 'age'), '')
        self.assertEqual(dictsortreversed(1, 'age'), '')

    def test_invalid_args(self):
        if False:
            i = 10
            return i + 15
        'Fail silently if invalid lookups are passed.'
        self.assertEqual(dictsortreversed([{}], '._private'), '')
        self.assertEqual(dictsortreversed([{'_private': 'test'}], '_private'), '')
        self.assertEqual(dictsortreversed([{'nested': {'_private': 'test'}}], 'nested._private'), '')