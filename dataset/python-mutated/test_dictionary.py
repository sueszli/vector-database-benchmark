from helium._impl.util.dictionary import inverse
from unittest import TestCase

class InverseTest(TestCase):

    def test_inverse_empty(self):
        if False:
            print('Hello World!')
        self.assertEqual({}, inverse({}))

    def test_inverse(self):
        if False:
            for i in range(10):
                print('nop')
        names_for_ints = {0: {'zero', 'naught'}, 1: {'one'}}
        ints_for_names = {'zero': {0}, 'naught': {0}, 'one': {1}}
        self.assertEqual(ints_for_names, inverse(names_for_ints))