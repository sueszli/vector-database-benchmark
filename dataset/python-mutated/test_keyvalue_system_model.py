from __future__ import absolute_import
import unittest2
from st2common.models.system.keyvalue import InvalidUserKeyReferenceError
from st2common.models.system.keyvalue import UserKeyReference

class UserKeyReferenceSystemModelTest(unittest2.TestCase):

    def test_to_string_reference(self):
        if False:
            return 10
        key_ref = UserKeyReference.to_string_reference(user='stanley', name='foo')
        self.assertEqual(key_ref, 'stanley:foo')
        self.assertRaises(ValueError, UserKeyReference.to_string_reference, user=None, name='foo')

    def test_from_string_reference(self):
        if False:
            print('Hello World!')
        (user, name) = UserKeyReference.from_string_reference('stanley:foo')
        self.assertEqual(user, 'stanley')
        self.assertEqual(name, 'foo')
        self.assertRaises(InvalidUserKeyReferenceError, UserKeyReference.from_string_reference, 'this_key_has_no_sep')