"""Unit tests for jobs.decorators.validation_decorators."""
from __future__ import annotations
import collections
import re
from core.jobs.decorators import validation_decorators
from core.jobs.types import model_property
from core.platform import models
from core.tests import test_utils
import apache_beam as beam
from typing import Dict, Final, FrozenSet, Iterator, List, Set, Tuple, Type
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import datastore_services
    from mypy_imports import exp_models
(base_models, exp_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.EXPLORATION])
datastore_services = models.Registry.import_datastore_services()

class MockAuditsExisting(validation_decorators.AuditsExisting):
    """Subclassed with overrides to avoid modifying the real decorator."""
    _DO_FN_TYPES_BY_KIND: Dict[str, Set[Type[beam.DoFn]]] = collections.defaultdict(set)

    @classmethod
    def get_audit_do_fn_types(cls, kind: str) -> FrozenSet[Type[beam.DoFn]]:
        if False:
            while True:
                i = 10
        'Test-only helper for getting the DoFns of a specific kind of a model.\n\n        Args:\n            kind: str. The kind of model.\n\n        Returns:\n            frozenset(type(DoFn)). The set of audits.\n        '
        return frozenset(cls._DO_FN_TYPES_BY_KIND[kind])

    @classmethod
    def clear(cls) -> None:
        if False:
            while True:
                i = 10
        'Test-only helper method for clearing the decorator.'
        cls._DO_FN_TYPES_BY_KIND.clear()

class DoFn(beam.DoFn):
    """Simple DoFn that does nothing."""

    def process(self, unused_item: None) -> None:
        if False:
            while True:
                i = 10
        'Does nothing.'
        pass

class UnrelatedDoFn(beam.DoFn):
    """Simple DoFn that does nothing."""

    def process(self, unused_item: None) -> None:
        if False:
            print('Hello World!')
        'Does nothing.'
        pass

class DerivedDoFn(DoFn):
    """Simple DoFn that derives from another."""

    def process(self, unused_item: None) -> None:
        if False:
            return 10
        'Does nothing.'
        pass

class NotDoFn:
    """Class that does not inherit from DoFn."""

    def process(self, unused_item: None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Does nothing.'
        pass

class FooModel(base_models.BaseModel):
    """A model that is not registered in core.platform."""
    pass

class BarModel(base_models.BaseModel):
    """A model that holds a reference to a FooModel ID."""
    BAR_CONSTANT: Final = 1
    foo_id = datastore_services.StringProperty()

class BazModel(base_models.BaseModel):
    """A model that holds a reference to a BarModel ID and a FooModel ID."""
    foo_id = datastore_services.StringProperty()
    bar_id = datastore_services.StringProperty()

class AuditsExistingTests(test_utils.TestBase):

    def tearDown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        MockAuditsExisting.clear()

    def test_has_no_do_fns_by_default(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(MockAuditsExisting.get_audit_do_fn_types_by_kind(), {})

    def test_targets_every_subclass_when_a_base_model_is_targeted(self) -> None:
        if False:
            return 10
        self.assertIs(MockAuditsExisting(base_models.BaseModel)(DoFn), DoFn)
        self.assertItemsEqual(MockAuditsExisting.get_audit_do_fn_types_by_kind().items(), [(cls.__name__, {DoFn}) for cls in [base_models.BaseModel] + models.Registry.get_all_storage_model_classes()])

    def test_replaces_base_do_fn_when_derived_do_fn_is_added_later(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        MockAuditsExisting(base_models.BaseModel)(DoFn)
        MockAuditsExisting(base_models.BaseModel)(UnrelatedDoFn)
        self.assertItemsEqual(MockAuditsExisting.get_audit_do_fn_types('BaseModel'), [DoFn, UnrelatedDoFn])
        self.assertItemsEqual(MockAuditsExisting.get_audit_do_fn_types('ExplorationModel'), [DoFn, UnrelatedDoFn])
        MockAuditsExisting(exp_models.ExplorationModel)(DerivedDoFn)
        self.assertItemsEqual(MockAuditsExisting.get_audit_do_fn_types('BaseModel'), [DoFn, UnrelatedDoFn])
        self.assertItemsEqual(MockAuditsExisting.get_audit_do_fn_types('ExplorationModel'), [DerivedDoFn, UnrelatedDoFn])

    def test_keeps_derived_do_fn_when_base_do_fn_is_added_later(self) -> None:
        if False:
            i = 10
            return i + 15
        MockAuditsExisting(exp_models.ExplorationModel)(DerivedDoFn)
        MockAuditsExisting(exp_models.ExplorationModel)(UnrelatedDoFn)
        self.assertItemsEqual(MockAuditsExisting.get_audit_do_fn_types('BaseModel'), [])
        self.assertItemsEqual(MockAuditsExisting.get_audit_do_fn_types('ExplorationModel'), [DerivedDoFn, UnrelatedDoFn])
        MockAuditsExisting(base_models.BaseModel)(DoFn)
        self.assertItemsEqual(MockAuditsExisting.get_audit_do_fn_types('BaseModel'), [DoFn])
        self.assertItemsEqual(MockAuditsExisting.get_audit_do_fn_types('ExplorationModel'), [DerivedDoFn, UnrelatedDoFn])

    def test_does_not_register_duplicate_do_fns(self) -> None:
        if False:
            while True:
                i = 10
        MockAuditsExisting(base_models.BaseModel)(DoFn)
        self.assertItemsEqual(MockAuditsExisting.get_audit_do_fn_types('BaseModel'), [DoFn])
        MockAuditsExisting(base_models.BaseModel)(DoFn)
        self.assertItemsEqual(MockAuditsExisting.get_audit_do_fn_types('BaseModel'), [DoFn])

    def test_raises_value_error_when_given_no_args(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'Must target at least one model'):
            MockAuditsExisting()

    def test_raises_type_error_when_given_unregistered_model(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, re.escape('%r is not a model registered in core.platform' % FooModel)):
            MockAuditsExisting(FooModel)

    def test_raises_type_error_when_decorating_non_do_fn_class(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, '%r is not a subclass of DoFn' % NotDoFn):
            MockAuditsExisting(base_models.BaseModel)(NotDoFn)

class MockRelationshipsOf(validation_decorators.RelationshipsOf):
    """Subclassed with overrides to avoid modifying the real decorator."""
    _ID_REFERENCING_PROPERTIES: Dict[model_property.ModelProperty, Set[str]] = collections.defaultdict(set)

    @classmethod
    def clear(cls) -> None:
        if False:
            while True:
                i = 10
        'Test-only helper method for clearing the decorator.'
        cls._ID_REFERENCING_PROPERTIES.clear()

class RelationshipsOfTests(test_utils.TestBase):

    def tearDown(self) -> None:
        if False:
            return 10
        super().tearDown()
        MockRelationshipsOf.clear()

    def get_property_of(self, model_class: Type[base_models.BaseModel], property_name: str) -> model_property.ModelProperty:
        if False:
            i = 10
            return i + 15
        'Helper method to create a ModelProperty.\n\n        Args:\n            model_class: *. A subclass of BaseModel.\n            property_name: str. The name of the property to get.\n\n        Returns:\n            ModelProperty. An object that encodes both property and the model it\n            belongs to.\n        '
        return model_property.ModelProperty(model_class, getattr(model_class, property_name))

    def test_has_no_relationships_by_default(self) -> None:
        if False:
            return 10
        self.assertEqual(MockRelationshipsOf.get_id_referencing_properties_by_kind_of_possessor(), {})
        self.assertEqual(MockRelationshipsOf.get_all_model_kinds_referenced_by_properties(), set())

    def test_valid_relationship_generator(self) -> None:
        if False:
            while True:
                i = 10

        @MockRelationshipsOf(BarModel)
        def bar_model_relationships(model: Type[BarModel]) -> Iterator[Tuple[model_property.PropertyType, List[Type[base_models.BaseModel]]]]:
            if False:
                for i in range(10):
                    print('nop')
            'Defines the relationships of BarModel.'
            yield (model.foo_id, [FooModel])
        self.assertEqual(MockRelationshipsOf.get_id_referencing_properties_by_kind_of_possessor(), {'BarModel': ((self.get_property_of(BarModel, 'foo_id'), ('FooModel',)),)})
        self.assertEqual(MockRelationshipsOf.get_model_kind_references('BarModel', 'foo_id'), {'FooModel'})
        self.assertEqual(MockRelationshipsOf.get_all_model_kinds_referenced_by_properties(), {'FooModel'})

    def test_accepts_id_as_property(self) -> None:
        if False:
            return 10

        @MockRelationshipsOf(BarModel)
        def bar_model_relationships(model: Type[base_models.BaseModel]) -> Iterator[Tuple[model_property.PropertyType, List[Type[base_models.BaseModel]]]]:
            if False:
                print('Hello World!')
            'Defines the relationships of BarModel.'
            yield (model.id, [BazModel])
        self.assertEqual(MockRelationshipsOf.get_id_referencing_properties_by_kind_of_possessor(), {'BarModel': ((self.get_property_of(BarModel, 'id'), ('BazModel',)),)})
        self.assertEqual(MockRelationshipsOf.get_model_kind_references('BarModel', 'id'), {'BazModel'})
        self.assertEqual(MockRelationshipsOf.get_all_model_kinds_referenced_by_properties(), {'BazModel'})

    def test_rejects_values_that_are_not_types(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        foo_model = FooModel()
        with self.assertRaisesRegex(TypeError, 'is an instance, not a type'):
            MockRelationshipsOf(foo_model)

    def test_rejects_types_that_are_not_models(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, 'not a subclass of BaseModel'):
            MockRelationshipsOf(int)

    def test_rejects_relationship_generator_with_wrong_name(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'Please rename the function'):

            @MockRelationshipsOf(BarModel)
            def unused_bar_model_relationships(unused_model: Type[base_models.BaseModel]) -> Iterator[Tuple[model_property.PropertyType, List[Type[base_models.BaseModel]]]]:
                if False:
                    i = 10
                    return i + 15
                'Defines the relationships of BarModel.'
                yield (BarModel.foo_id, [FooModel])