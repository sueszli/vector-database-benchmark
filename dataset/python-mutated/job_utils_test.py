"""Unit tests for jobs.job_utils."""
from __future__ import annotations
import datetime
from core import feconf
from core.jobs import job_utils
from core.platform import models
from core.tests import test_utils
from apache_beam.io.gcp.datastore.v1new import types as beam_datastore_types
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import datastore_services
(base_models,) = models.Registry.import_models([models.Names.BASE_MODEL])
datastore_services = models.Registry.import_datastore_services()

class FooModel(datastore_services.Model):
    """Simple BaseModel subclass with a 'prop' string property."""
    prop = datastore_services.StringProperty()

class BarModel(datastore_services.Model):
    """Simple BaseModel subclass with a 'prop' integer property."""
    prop = datastore_services.IntegerProperty()

class CoreModel(base_models.BaseModel):
    """Simple BaseModel subclass with a 'prop' float property."""
    prop = datastore_services.FloatProperty()

class CloneTests(test_utils.TestBase):

    def test_clone_model(self) -> None:
        if False:
            return 10
        model = base_models.BaseModel(id='123', deleted=True)
        clone = job_utils.clone_model(model)
        self.assertEqual(model.id, clone.id)
        self.assertEqual(model, clone)
        self.assertIsNot(model, clone)
        self.assertIsInstance(clone, base_models.BaseModel)

    def test_clone_with_changes(self) -> None:
        if False:
            print('Hello World!')
        model = base_models.BaseModel(id='123', deleted=True)
        clone = job_utils.clone_model(model, deleted=False)
        self.assertNotEqual(model, clone)
        self.assertIsNot(model, clone)
        self.assertIsInstance(clone, base_models.BaseModel)
        self.assertTrue(model.deleted)
        self.assertFalse(clone.deleted)

    def test_clone_with_changes_to_id(self) -> None:
        if False:
            print('Hello World!')
        model = base_models.BaseModel(id='123')
        clone = job_utils.clone_model(model, id='124')
        self.assertNotEqual(model, clone)
        self.assertIsNot(model, clone)
        self.assertIsInstance(clone, base_models.BaseModel)
        self.assertEqual(model.id, '123')
        self.assertEqual(clone.id, '124')

    def test_clone_sub_class(self) -> None:
        if False:
            print('Hello World!')
        model = FooModel(prop='original')
        clone = job_utils.clone_model(model)
        self.assertEqual(model, clone)
        self.assertIsNot(model, clone)
        self.assertIsInstance(clone, FooModel)
        self.assertEqual(model.prop, 'original')
        self.assertEqual(clone.prop, 'original')

    def test_clone_sub_class_with_changes(self) -> None:
        if False:
            i = 10
            return i + 15
        model = FooModel(prop='original')
        clone = job_utils.clone_model(model, prop='updated')
        self.assertNotEqual(model, clone)
        self.assertIsNot(model, clone)
        self.assertIsInstance(clone, FooModel)
        self.assertEqual(model.prop, 'original')
        self.assertEqual(clone.prop, 'updated')

class GetModelClassTests(test_utils.TestBase):

    def test_get_from_existing_model(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertIs(job_utils.get_model_class('BaseModel'), base_models.BaseModel)

    def test_get_from_non_existing_model(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(Exception, 'No model class found'):
            job_utils.get_model_class('InvalidModel')

class GetModelKindTests(test_utils.TestBase):

    def test_get_from_datastore_model(self) -> None:
        if False:
            while True:
                i = 10
        model = base_models.BaseModel()
        self.assertEqual(job_utils.get_model_kind(model), 'BaseModel')

    def test_get_from_datastore_model_class(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(job_utils.get_model_kind(base_models.BaseModel), 'BaseModel')

    def test_get_from_bad_value(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(TypeError, 'not a model type or instance'):
            job_utils.get_model_kind(123)

class GetModelPropertyTests(test_utils.TestBase):

    def test_get_id_from_datastore_model(self) -> None:
        if False:
            while True:
                i = 10
        model = FooModel(id='123')
        self.assertEqual(job_utils.get_model_property(model, 'id'), '123')

    def test_get_property_from_datastore_model(self) -> None:
        if False:
            return 10
        model = FooModel(prop='abc')
        self.assertEqual(job_utils.get_model_property(model, 'prop'), 'abc')

    def test_get_missing_property_from_datastore_model(self) -> None:
        if False:
            return 10
        model = FooModel()
        self.assertEqual(job_utils.get_model_property(model, 'prop'), None)

    def test_get_property_from_bad_value(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, 'not a model instance'):
            job_utils.get_model_property(123, 'prop')

class GetModelIdTests(test_utils.TestBase):

    def test_get_id_from_datastore_model(self) -> None:
        if False:
            while True:
                i = 10
        model = FooModel(id='123')
        self.assertEqual(job_utils.get_model_id(model), '123')

    def test_get_id_from_bad_value(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, 'not a model instance'):
            job_utils.get_model_id(123)

class BeamEntityToAndFromModelTests(test_utils.TestBase):

    def test_get_beam_entity_from_model(self) -> None:
        if False:
            return 10
        model = FooModel(id='abc', project=feconf.OPPIA_PROJECT_ID, prop='123')
        beam_entity = job_utils.get_beam_entity_from_ndb_model(model)
        self.assertEqual(beam_entity.key.path_elements, ('FooModel', 'abc'))
        self.assertEqual(beam_entity.key.project, feconf.OPPIA_PROJECT_ID)
        self.assertEqual(beam_entity.properties, {'prop': '123'})

    def test_get_model_from_beam_entity(self) -> None:
        if False:
            while True:
                i = 10
        beam_entity = beam_datastore_types.Entity(beam_datastore_types.Key(('FooModel', 'abc'), project=feconf.OPPIA_PROJECT_ID, namespace=self.namespace))
        beam_entity.set_properties({'prop': '123'})
        self.assertEqual(FooModel(id='abc', project=feconf.OPPIA_PROJECT_ID, prop='123'), job_utils.get_ndb_model_from_beam_entity(beam_entity))

    def test_get_beam_key_from_ndb_key(self) -> None:
        if False:
            i = 10
            return i + 15
        beam_key = beam_datastore_types.Key(('FooModel', 'abc'), project=feconf.OPPIA_PROJECT_ID, namespace=self.namespace)
        ndb_key = datastore_services.Key._from_ds_key(beam_key.to_client_key())
        self.assertEqual(job_utils.get_beam_key_from_ndb_key(ndb_key), beam_key)

    def test_get_model_from_beam_entity_with_time(self) -> None:
        if False:
            return 10
        utcnow = datetime.datetime.utcnow()
        beam_entity = beam_datastore_types.Entity(beam_datastore_types.Key(('CoreModel', 'abc'), project=feconf.OPPIA_PROJECT_ID, namespace=self.namespace))
        beam_entity.set_properties({'prop': 3.14, 'created_on': utcnow.replace(tzinfo=datetime.timezone.utc), 'last_updated': None, 'deleted': False})
        self.assertEqual(CoreModel(id='abc', project=feconf.OPPIA_PROJECT_ID, prop=3.14, created_on=utcnow), job_utils.get_ndb_model_from_beam_entity(beam_entity))

    def test_from_and_then_to_model(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        model = FooModel(id='abc', project=feconf.OPPIA_PROJECT_ID, prop='123')
        self.assertEqual(model, job_utils.get_ndb_model_from_beam_entity(job_utils.get_beam_entity_from_ndb_model(model)))

    def test_from_and_then_to_beam_entity(self) -> None:
        if False:
            i = 10
            return i + 15
        beam_entity = beam_datastore_types.Entity(beam_datastore_types.Key(('CoreModel', 'abc'), project=feconf.OPPIA_PROJECT_ID))
        beam_entity.set_properties({'prop': 123, 'created_on': datetime.datetime.utcnow(), 'last_updated': datetime.datetime.utcnow(), 'deleted': False})
        self.assertEqual(beam_entity, job_utils.get_beam_entity_from_ndb_model(job_utils.get_ndb_model_from_beam_entity(beam_entity)))

class GetBeamQueryFromNdbQueryTests(test_utils.TestBase):

    def test_query_everything(self) -> None:
        if False:
            while True:
                i = 10
        query = datastore_services.query_everything()
        beam_query = job_utils.get_beam_query_from_ndb_query(query)
        self.assertIsNone(beam_query.kind)
        self.assertEqual(beam_query.order, ('__key__',))

    def test_query_with_kind(self) -> None:
        if False:
            i = 10
            return i + 15
        query = base_models.BaseModel.query()
        beam_query = job_utils.get_beam_query_from_ndb_query(query)
        self.assertEqual(beam_query.kind, 'BaseModel')

    def test_query_with_namespace(self) -> None:
        if False:
            return 10
        query = datastore_services.Query(namespace='abc')
        beam_query = job_utils.get_beam_query_from_ndb_query(query)
        self.assertEqual(beam_query.namespace, 'abc')

    def test_query_with_filter(self) -> None:
        if False:
            while True:
                i = 10
        query = datastore_services.Query(filters=BarModel.prop >= 3)
        beam_query = job_utils.get_beam_query_from_ndb_query(query)
        self.assertEqual(beam_query.filters, (('prop', '>=', 3),))

    def test_query_with_range_like_filter(self) -> None:
        if False:
            return 10
        query = datastore_services.Query(filters=datastore_services.all_of(BarModel.prop >= 3, BarModel.prop < 6))
        beam_query = job_utils.get_beam_query_from_ndb_query(query)
        self.assertEqual(beam_query.filters, (('prop', '>=', 3), ('prop', '<', 6)))

    def test_query_with_or_filter_raises_type_error(self) -> None:
        if False:
            return 10
        query = datastore_services.Query(filters=datastore_services.any_of(BarModel.prop == 1, BarModel.prop == 2))
        with self.assertRaisesRegex(TypeError, 'forbidden filter'):
            job_utils.get_beam_query_from_ndb_query(query)

    def test_query_with_in_filter_raises_type_error(self) -> None:
        if False:
            return 10
        query = datastore_services.Query(filters=BarModel.prop.IN([1, 2, 3]))
        with self.assertRaisesRegex(TypeError, 'forbidden filter'):
            job_utils.get_beam_query_from_ndb_query(query)

    def test_query_with_not_equal_filter_raises_type_error(self) -> None:
        if False:
            print('Hello World!')
        query = datastore_services.Query(filters=BarModel.prop != 1)
        with self.assertRaisesRegex(TypeError, 'forbidden filter'):
            job_utils.get_beam_query_from_ndb_query(query)

    def test_query_with_order(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        query = BarModel.query().order(BarModel.prop)
        beam_query = job_utils.get_beam_query_from_ndb_query(query)
        self.assertEqual(beam_query.order, ('prop',))

    def test_query_with_multiple_orders(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        query = BarModel.query().order(BarModel.prop, BarModel.prop)
        beam_query = job_utils.get_beam_query_from_ndb_query(query)
        self.assertEqual(beam_query.order, ('prop', 'prop'))

    def test_query_with_descending_order(self) -> None:
        if False:
            print('Hello World!')
        query = BarModel.query().order(-BarModel.prop)
        beam_query = job_utils.get_beam_query_from_ndb_query(query)
        self.assertEqual(beam_query.order, ('-prop',))