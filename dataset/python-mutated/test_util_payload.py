from __future__ import absolute_import
import unittest2
from st2common.util.payload import PayloadLookup
__all__ = ['PayloadLookupTestCase']

class PayloadLookupTestCase(unittest2.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.payload = PayloadLookup({'pikachu': 'Has no ears', 'charmander': 'Plays with fire'})
        super(PayloadLookupTestCase, cls).setUpClass()

    def test_get_key(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.payload.get_value('trigger.pikachu'), ['Has no ears'])
        self.assertEqual(self.payload.get_value('trigger.charmander'), ['Plays with fire'])

    def test_explicitly_get_multiple_keys(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.payload.get_value('trigger.pikachu[*]'), ['Has no ears'])
        self.assertEqual(self.payload.get_value('trigger.charmander[*]'), ['Plays with fire'])

    def test_get_nonexistent_key(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(self.payload.get_value('trigger.squirtle'))