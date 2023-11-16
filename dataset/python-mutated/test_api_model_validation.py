from __future__ import absolute_import
import unittest2
from st2common.models.api.base import BaseAPI
__all__ = ['APIModelValidationTestCase']

class MockAPIModel1(BaseAPI):
    model = None
    schema = {'title': 'MockAPIModel', 'description': 'Test', 'type': 'object', 'properties': {'id': {'description': 'The unique identifier for the action runner.', 'type': ['string', 'null'], 'default': None}, 'name': {'description': 'The name of the action runner.', 'type': 'string', 'required': True}, 'description': {'description': 'The description of the action runner.', 'type': 'string'}, 'enabled': {'type': 'boolean', 'default': True}, 'parameters': {'type': 'object'}, 'permission_grants': {'type': 'array', 'items': {'type': 'object', 'properties': {'resource_uid': {'type': 'string', 'description': 'UID of a resource to which this grant applies to.', 'required': False, 'default': 'unknown'}, 'enabled': {'type': 'boolean', 'default': True}, 'description': {'type': 'string', 'description': 'Description', 'required': False}}}, 'default': []}}, 'additionalProperties': False}

class MockAPIModel2(BaseAPI):
    model = None
    schema = {'title': 'MockAPIModel2', 'description': 'Test', 'type': 'object', 'properties': {'id': {'description': 'The unique identifier for the action runner.', 'type': 'string', 'default': None}, 'permission_grants': {'type': 'array', 'items': {'type': 'object', 'properties': {'resource_uid': {'type': 'string', 'description': 'UID of a resource to which this grant applies to.', 'required': False, 'default': None}, 'description': {'type': 'string', 'required': True}}}, 'default': []}, 'parameters': {'type': 'object', 'properties': {'id': {'type': 'string', 'default': None}, 'name': {'type': 'string', 'required': True}}, 'additionalProperties': False}}, 'additionalProperties': False}

class APIModelValidationTestCase(unittest2.TestCase):

    def test_validate_default_values_are_set(self):
        if False:
            for i in range(10):
                print('nop')
        mock_model_api = MockAPIModel1(name='name')
        self.assertEqual(getattr(mock_model_api, 'id', 'notset'), 'notset')
        self.assertEqual(mock_model_api.name, 'name')
        self.assertEqual(getattr(mock_model_api, 'enabled', None), None)
        self.assertEqual(getattr(mock_model_api, 'permission_grants', None), None)
        mock_model_api_validated = mock_model_api.validate()
        self.assertEqual(getattr(mock_model_api, 'id', 'notset'), 'notset')
        self.assertEqual(mock_model_api.name, 'name')
        self.assertEqual(getattr(mock_model_api, 'enabled', None), None)
        self.assertEqual(mock_model_api_validated.id, None)
        self.assertEqual(mock_model_api_validated.name, 'name')
        self.assertEqual(mock_model_api_validated.enabled, True)
        self.assertEqual(mock_model_api_validated.permission_grants, [])
        mock_model_api = MockAPIModel1(name='name', enabled=False, permission_grants=[{}, {'description': 'test'}])
        self.assertEqual(mock_model_api.name, 'name')
        self.assertEqual(mock_model_api.enabled, False)
        self.assertEqual(mock_model_api.permission_grants, [{}, {'description': 'test'}])
        mock_model_api_validated = mock_model_api.validate()
        self.assertEqual(mock_model_api.name, 'name')
        self.assertEqual(mock_model_api.enabled, False)
        self.assertEqual(mock_model_api.permission_grants, [{}, {'description': 'test'}])
        self.assertEqual(mock_model_api_validated.id, None)
        self.assertEqual(mock_model_api_validated.name, 'name')
        self.assertEqual(mock_model_api_validated.enabled, False)
        self.assertEqual(mock_model_api_validated.permission_grants, [{'resource_uid': 'unknown', 'enabled': True}, {'resource_uid': 'unknown', 'enabled': True, 'description': 'test'}])

    def test_validate_nested_attribute_with_default_not_provided(self):
        if False:
            print('Hello World!')
        mock_model_api = MockAPIModel2()
        self.assertEqual(getattr(mock_model_api, 'id', 'notset'), 'notset')
        self.assertEqual(getattr(mock_model_api, 'permission_grants', 'notset'), 'notset')
        self.assertEqual(getattr(mock_model_api, 'parameters', 'notset'), 'notset')
        mock_model_api_validated = mock_model_api.validate()
        self.assertEqual(getattr(mock_model_api, 'id', 'notset'), 'notset')
        self.assertEqual(getattr(mock_model_api, 'permission_grants', 'notset'), 'notset')
        self.assertEqual(getattr(mock_model_api, 'parameters', 'notset'), 'notset')
        self.assertEqual(mock_model_api_validated.id, None)
        self.assertEqual(mock_model_api_validated.permission_grants, [])
        self.assertEqual(getattr(mock_model_api_validated, 'parameters', 'notset'), 'notset')

    def test_validate_allow_default_none_for_any_type(self):
        if False:
            while True:
                i = 10
        mock_model_api = MockAPIModel2(permission_grants=[{'description': 'test'}], parameters={'name': 'test'})
        self.assertEqual(getattr(mock_model_api, 'id', 'notset'), 'notset')
        self.assertEqual(mock_model_api.permission_grants, [{'description': 'test'}])
        self.assertEqual(mock_model_api.parameters, {'name': 'test'})
        mock_model_api_validated = mock_model_api.validate()
        self.assertEqual(getattr(mock_model_api, 'id', 'notset'), 'notset')
        self.assertEqual(mock_model_api.permission_grants, [{'description': 'test'}])
        self.assertEqual(mock_model_api.parameters, {'name': 'test'})
        self.assertEqual(mock_model_api_validated.id, None)
        self.assertEqual(mock_model_api_validated.permission_grants, [{'description': 'test', 'resource_uid': None}])
        self.assertEqual(mock_model_api_validated.parameters, {'id': None, 'name': 'test'})