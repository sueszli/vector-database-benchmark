"""Tests for methods in the action registry."""
from __future__ import annotations
from core.domain import activity_domain
from core.tests import test_utils

class ActivityReferenceDomainUnitTests(test_utils.GenericTestBase):
    """Tests for ActivityReference domain class."""

    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        self.exp_activity_reference = activity_domain.ActivityReference('exploration', '1234')
        self.collection_activity_reference = activity_domain.ActivityReference('collection', '1234')
        self.invalid_activity_reference_with_invalid_type = activity_domain.ActivityReference('invalid_activity_type', '1234')

    def test_that_hashes_for_different_object_types_are_distinct(self) -> None:
        if False:
            print('Hello World!')
        exp_hash = self.exp_activity_reference.get_hash()
        collection_hash = self.collection_activity_reference.get_hash()
        invalid_activity_hash = self.invalid_activity_reference_with_invalid_type.get_hash()
        self.assertNotEqual(exp_hash, collection_hash)
        self.assertNotEqual(exp_hash, invalid_activity_hash)
        self.assertNotEqual(collection_hash, invalid_activity_hash)

    def test_validate_with_invalid_type(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(Exception, 'Invalid activity type: invalid_activity_type'):
            self.invalid_activity_reference_with_invalid_type.validate()

    def test_validate_with_invalid_id(self) -> None:
        if False:
            print('Hello World!')
        invalid_activity_reference_with_invalid_id = activity_domain.ActivityReference('exploration', 1234)
        with self.assertRaisesRegex(Exception, 'Expected id to be a string but found 1234'):
            invalid_activity_reference_with_invalid_id.validate()

    def test_to_dict(self) -> None:
        if False:
            i = 10
            return i + 15
        exp_dict = self.exp_activity_reference.to_dict()
        collection_dict = self.collection_activity_reference.to_dict()
        self.assertEqual(exp_dict, {'type': 'exploration', 'id': '1234'})
        self.assertEqual(collection_dict, {'type': 'collection', 'id': '1234'})

    def test_from_dict(self) -> None:
        if False:
            i = 10
            return i + 15
        sample_dict = {'type': 'exploration', 'id': '1234'}
        returned_activity_object = activity_domain.ActivityReference.from_dict(sample_dict)
        returned_activity_dict = returned_activity_object.to_dict()
        self.assertEqual(sample_dict, returned_activity_dict)
        self.assertEqual(sample_dict['type'], returned_activity_dict['type'])
        self.assertEqual(sample_dict['id'], returned_activity_dict['id'])

class ActivityReferencesDomainUnitTests(test_utils.GenericTestBase):
    """Tests for ActivityReferences domain class."""

    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        exp_activity_reference = activity_domain.ActivityReference('exploration', '1234')
        collection_activity_reference = activity_domain.ActivityReference('collection', '1234')
        invalid_activity_reference = activity_domain.ActivityReference('invalid_activity_type', '1234')
        self.valid_activity_references = activity_domain.ActivityReferences([exp_activity_reference, collection_activity_reference])
        self.invalid_activity_references = activity_domain.ActivityReferences([exp_activity_reference, invalid_activity_reference])

    def test_validate_passes_with_valid_activity_reference_list(self) -> None:
        if False:
            while True:
                i = 10
        self.valid_activity_references.validate()

    def test_validate_fails_with_invalid_type_in_activity_reference_list(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(Exception, 'Invalid activity type: invalid_activity_type'):
            self.invalid_activity_references.validate()