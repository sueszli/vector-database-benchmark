"""Unit tests for jobs.transforms.base_validation_registry."""
from __future__ import annotations
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import base_validation_registry
from core.tests import test_utils

class GetAuditsByKindTests(test_utils.TestBase):
    unique_obj = object()

    @classmethod
    def get_audit_do_fn_types_by_kind_mock(cls) -> object:
        if False:
            return 10
        'Returns the unique_obj.'
        return cls.unique_obj

    def test_returns_value_from_decorator(self) -> None:
        if False:
            while True:
                i = 10
        get_audit_do_fn_types_by_kind_swap = self.swap(validation_decorators.AuditsExisting, 'get_audit_do_fn_types_by_kind', self.get_audit_do_fn_types_by_kind_mock)
        with get_audit_do_fn_types_by_kind_swap:
            self.assertIs(base_validation_registry.get_audit_do_fn_types_by_kind(), self.unique_obj)

class GetIdReferencingPropertiesByKindOfPossessorTests(test_utils.TestBase):
    unique_obj = object()

    @classmethod
    def get_id_referencing_properties_by_kind_of_possessor_mock(cls) -> object:
        if False:
            i = 10
            return i + 15
        'Returns the unique_obj.'
        return cls.unique_obj

    def test_returns_value_from_decorator(self) -> None:
        if False:
            while True:
                i = 10
        get_id_referencing_properties_by_kind_of_possessor_swap = self.swap(validation_decorators.RelationshipsOf, 'get_id_referencing_properties_by_kind_of_possessor', self.get_id_referencing_properties_by_kind_of_possessor_mock)
        with get_id_referencing_properties_by_kind_of_possessor_swap:
            self.assertIs(base_validation_registry.get_id_referencing_properties_by_kind_of_possessor(), self.unique_obj)

class GetAllModelKindsReferencedByPropertiesTests(test_utils.TestBase):
    unique_obj = object()

    @classmethod
    def get_all_model_kinds_referenced_by_properties_mock(cls) -> object:
        if False:
            print('Hello World!')
        'Returns the unique_obj.'
        return cls.unique_obj

    def test_returns_value_from_decorator(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        get_all_model_kinds_referenced_by_properties_swap = self.swap(validation_decorators.RelationshipsOf, 'get_all_model_kinds_referenced_by_properties', self.get_all_model_kinds_referenced_by_properties_mock)
        with get_all_model_kinds_referenced_by_properties_swap:
            self.assertIs(base_validation_registry.get_all_model_kinds_referenced_by_properties(), self.unique_obj)