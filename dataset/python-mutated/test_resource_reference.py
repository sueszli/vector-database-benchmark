from __future__ import absolute_import
import unittest2
from st2common.models.system.common import ResourceReference
from st2common.models.system.common import InvalidResourceReferenceError

class ResourceReferenceTestCase(unittest2.TestCase):

    def test_resource_reference_success(self):
        if False:
            for i in range(10):
                print('nop')
        value = 'pack1.name1'
        ref = ResourceReference.from_string_reference(ref=value)
        self.assertEqual(ref.pack, 'pack1')
        self.assertEqual(ref.name, 'name1')
        self.assertEqual(ref.ref, value)
        ref = ResourceReference(pack='pack1', name='name1')
        self.assertEqual(ref.ref, 'pack1.name1')
        ref = ResourceReference(pack='pack1', name='name1.name2')
        self.assertEqual(ref.ref, 'pack1.name1.name2')

    def test_resource_reference_failure(self):
        if False:
            return 10
        self.assertRaises(InvalidResourceReferenceError, ResourceReference.from_string_reference, ref='blah')
        self.assertRaises(InvalidResourceReferenceError, ResourceReference.from_string_reference, ref=None)

    def test_to_string_reference(self):
        if False:
            while True:
                i = 10
        ref = ResourceReference.to_string_reference(pack='mapack', name='moname')
        self.assertEqual(ref, 'mapack.moname')
        expected_msg = 'Pack name should not contain "\\."'
        self.assertRaisesRegexp(ValueError, expected_msg, ResourceReference.to_string_reference, pack='pack.invalid', name='bar')
        expected_msg = 'Both pack and name needed for building'
        self.assertRaisesRegexp(ValueError, expected_msg, ResourceReference.to_string_reference, pack='pack', name=None)
        expected_msg = 'Both pack and name needed for building'
        self.assertRaisesRegexp(ValueError, expected_msg, ResourceReference.to_string_reference, pack=None, name='name')

    def test_is_resource_reference(self):
        if False:
            print('Hello World!')
        self.assertTrue(ResourceReference.is_resource_reference('foo.bar'))
        self.assertTrue(ResourceReference.is_resource_reference('foo.bar.ponies'))
        self.assertFalse(ResourceReference.is_resource_reference('foo'))