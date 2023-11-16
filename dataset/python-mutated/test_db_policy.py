from __future__ import absolute_import
import unittest2
from st2common.constants import pack as pack_constants
from st2common.models.db.policy import PolicyTypeReference, PolicyTypeDB, PolicyDB
from st2common.models.system.common import InvalidReferenceError
from st2common.persistence.policy import PolicyType, Policy
from st2tests import DbModelTestCase

class PolicyTypeReferenceTest(unittest2.TestCase):

    def test_is_reference(self):
        if False:
            print('Hello World!')
        self.assertTrue(PolicyTypeReference.is_reference('action.concurrency'))
        self.assertFalse(PolicyTypeReference.is_reference('concurrency'))
        self.assertFalse(PolicyTypeReference.is_reference(''))
        self.assertFalse(PolicyTypeReference.is_reference(None))

    def test_validate_resource_type(self):
        if False:
            print('Hello World!')
        self.assertEqual(PolicyTypeReference.validate_resource_type('action'), 'action')
        self.assertRaises(ValueError, PolicyTypeReference.validate_resource_type, 'action.test')

    def test_get_resource_type(self):
        if False:
            while True:
                i = 10
        self.assertEqual(PolicyTypeReference.get_resource_type('action.concurrency'), 'action')
        self.assertRaises(InvalidReferenceError, PolicyTypeReference.get_resource_type, '.abc')
        self.assertRaises(InvalidReferenceError, PolicyTypeReference.get_resource_type, 'abc')
        self.assertRaises(InvalidReferenceError, PolicyTypeReference.get_resource_type, '')
        self.assertRaises(InvalidReferenceError, PolicyTypeReference.get_resource_type, None)

    def test_get_name(self):
        if False:
            print('Hello World!')
        self.assertEqual(PolicyTypeReference.get_name('action.concurrency'), 'concurrency')
        self.assertRaises(InvalidReferenceError, PolicyTypeReference.get_name, '.abc')
        self.assertRaises(InvalidReferenceError, PolicyTypeReference.get_name, 'abc')
        self.assertRaises(InvalidReferenceError, PolicyTypeReference.get_name, '')
        self.assertRaises(InvalidReferenceError, PolicyTypeReference.get_name, None)

    def test_to_string_reference(self):
        if False:
            while True:
                i = 10
        ref = PolicyTypeReference.to_string_reference(resource_type='action', name='concurrency')
        self.assertEqual(ref, 'action.concurrency')
        self.assertRaises(ValueError, PolicyTypeReference.to_string_reference, resource_type='action.test', name='concurrency')
        self.assertRaises(ValueError, PolicyTypeReference.to_string_reference, resource_type=None, name='concurrency')
        self.assertRaises(ValueError, PolicyTypeReference.to_string_reference, resource_type='', name='concurrency')
        self.assertRaises(ValueError, PolicyTypeReference.to_string_reference, resource_type='action', name=None)
        self.assertRaises(ValueError, PolicyTypeReference.to_string_reference, resource_type='action', name='')
        self.assertRaises(ValueError, PolicyTypeReference.to_string_reference, resource_type=None, name=None)
        self.assertRaises(ValueError, PolicyTypeReference.to_string_reference, resource_type='', name='')

    def test_from_string_reference(self):
        if False:
            return 10
        ref = PolicyTypeReference.from_string_reference('action.concurrency')
        self.assertEqual(ref.resource_type, 'action')
        self.assertEqual(ref.name, 'concurrency')
        self.assertEqual(ref.ref, 'action.concurrency')
        ref = PolicyTypeReference.from_string_reference('action.concurrency.targeted')
        self.assertEqual(ref.resource_type, 'action')
        self.assertEqual(ref.name, 'concurrency.targeted')
        self.assertEqual(ref.ref, 'action.concurrency.targeted')
        self.assertRaises(InvalidReferenceError, PolicyTypeReference.from_string_reference, '.test')
        self.assertRaises(InvalidReferenceError, PolicyTypeReference.from_string_reference, '')
        self.assertRaises(InvalidReferenceError, PolicyTypeReference.from_string_reference, None)

class PolicyTypeTest(DbModelTestCase):
    access_type = PolicyType

    @staticmethod
    def _create_instance():
        if False:
            print('Hello World!')
        parameters = {'threshold': {'type': 'integer', 'required': True}}
        instance = PolicyTypeDB(name='concurrency', description='TBD', enabled=None, ref=None, resource_type='action', module='st2action.policies.concurrency', parameters=parameters)
        return instance

    def test_crud(self):
        if False:
            print('Hello World!')
        instance = self._create_instance()
        defaults = {'ref': 'action.concurrency', 'enabled': True}
        updates = {'description': 'Limits the concurrent executions for the action.'}
        self._assert_crud(instance, defaults=defaults, updates=updates)

    def test_unique_key(self):
        if False:
            while True:
                i = 10
        instance = self._create_instance()
        self._assert_unique_key_constraint(instance)

class PolicyTest(DbModelTestCase):
    access_type = Policy

    @staticmethod
    def _create_instance():
        if False:
            return 10
        instance = PolicyDB(pack=None, name='local.concurrency', description='TBD', enabled=None, ref=None, resource_ref='core.local', policy_type='action.concurrency', parameters={'threshold': 25})
        return instance

    def test_crud(self):
        if False:
            i = 10
            return i + 15
        instance = self._create_instance()
        defaults = {'pack': pack_constants.DEFAULT_PACK_NAME, 'ref': '%s.local.concurrency' % pack_constants.DEFAULT_PACK_NAME, 'enabled': True}
        updates = {'description': 'Limits the concurrent executions for the action "core.local".'}
        self._assert_crud(instance, defaults=defaults, updates=updates)

    def test_ref(self):
        if False:
            i = 10
            return i + 15
        instance = self._create_instance()
        ref = instance.get_reference()
        self.assertIsNotNone(ref)
        self.assertEqual(ref.pack, instance.pack)
        self.assertEqual(ref.name, instance.name)
        self.assertEqual(ref.ref, instance.pack + '.' + instance.name)
        self.assertEqual(ref.ref, instance.ref)

    def test_unique_key(self):
        if False:
            while True:
                i = 10
        instance = self._create_instance()
        self._assert_unique_key_constraint(instance)