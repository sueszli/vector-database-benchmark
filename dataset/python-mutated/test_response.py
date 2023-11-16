import pytest
from boto3.resources.base import ResourceMeta, ServiceResource
from boto3.resources.factory import ResourceFactory
from boto3.resources.model import Parameter, ResponseResource
from boto3.resources.response import RawHandler, ResourceHandler, build_empty_response, build_identifiers
from boto3.utils import ServiceContext
from tests import BaseTestCase, mock

class TestBuildIdentifiers(BaseTestCase):

    def test_build_identifier_from_res_path_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        identifiers = [Parameter(target='Id', source='response', path='Container.Frob.Id')]
        parent = mock.Mock()
        params = {}
        response = {'Container': {'Frob': {'Id': 'response-path'}}}
        values = build_identifiers(identifiers, parent, params, response)
        assert values[0][1] == 'response-path'

    def test_build_identifier_from_res_path_list(self):
        if False:
            return 10
        identifiers = [Parameter(target='Id', source='response', path='Container.Frobs[].Id')]
        parent = mock.Mock()
        params = {}
        response = {'Container': {'Frobs': [{'Id': 'response-path'}]}}
        values = build_identifiers(identifiers, parent, params, response)
        assert values[0][1] == ['response-path']

    def test_build_identifier_from_parent_identifier(self):
        if False:
            print('Hello World!')
        identifiers = [Parameter(target='Id', source='identifier', name='Id')]
        parent = mock.Mock()
        parent.id = 'identifier'
        params = {}
        response = {'Container': {'Frobs': []}}
        values = build_identifiers(identifiers, parent, params, response)
        assert values[0][1] == 'identifier'

    def test_build_identifier_from_parent_data_member(self):
        if False:
            while True:
                i = 10
        identifiers = [Parameter(target='Id', source='data', path='Member')]
        parent = mock.Mock()
        parent.meta = ResourceMeta('test', data={'Member': 'data-member'})
        params = {}
        response = {'Container': {'Frobs': []}}
        values = build_identifiers(identifiers, parent, params, response)
        assert values[0][1] == 'data-member'

    def test_build_identifier_from_req_param(self):
        if False:
            i = 10
            return i + 15
        identifiers = [Parameter(target='Id', source='requestParameter', path='Param')]
        parent = mock.Mock()
        params = {'Param': 'request-param'}
        response = {'Container': {'Frobs': []}}
        values = build_identifiers(identifiers, parent, params, response)
        assert values[0][1] == 'request-param'

    def test_build_identifier_from_invalid_source_type(self):
        if False:
            print('Hello World!')
        identifiers = [Parameter(target='Id', source='invalid')]
        parent = mock.Mock()
        params = {}
        response = {'Container': {'Frobs': []}}
        with pytest.raises(NotImplementedError):
            build_identifiers(identifiers, parent, params, response)

class TestBuildEmptyResponse(BaseTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.search_path = ''
        self.operation_name = 'GetFrobs'
        self.output_shape = mock.Mock()
        operation_model = mock.Mock()
        operation_model.output_shape = self.output_shape
        self.service_model = mock.Mock()
        self.service_model.operation_model.return_value = operation_model

    def get_response(self):
        if False:
            i = 10
            return i + 15
        return build_empty_response(self.search_path, self.operation_name, self.service_model)

    def test_empty_structure(self):
        if False:
            for i in range(10):
                print('nop')
        self.output_shape.type_name = 'structure'
        response = self.get_response()
        assert isinstance(response, dict)
        assert response == {}

    def test_empty_list(self):
        if False:
            print('Hello World!')
        self.output_shape.type_name = 'list'
        response = self.get_response()
        assert isinstance(response, list)
        assert len(response) == 0

    def test_empty_map(self):
        if False:
            for i in range(10):
                print('nop')
        self.output_shape.type_name = 'map'
        response = self.get_response()
        assert isinstance(response, dict)
        assert response == {}

    def test_empty_string(self):
        if False:
            print('Hello World!')
        self.output_shape.type_name = 'string'
        response = self.get_response()
        assert response is None

    def test_empty_integer(self):
        if False:
            i = 10
            return i + 15
        self.output_shape.type_name = 'integer'
        response = self.get_response()
        assert response is None

    def test_empty_unknown_returns_none(self):
        if False:
            return 10
        self.output_shape.type_name = 'invalid'
        response = self.get_response()
        assert response is None

    def test_path_structure(self):
        if False:
            print('Hello World!')
        self.search_path = 'Container.Frob'
        frob = mock.Mock()
        frob.type_name = 'integer'
        container = mock.Mock()
        container.type_name = 'structure'
        container.members = {'Frob': frob}
        self.output_shape.type_name = 'structure'
        self.output_shape.members = {'Container': container}
        response = self.get_response()
        assert response is None

    def test_path_list(self):
        if False:
            return 10
        self.search_path = 'Container[1].Frob'
        frob = mock.Mock()
        frob.type_name = 'integer'
        container = mock.Mock()
        container.type_name = 'list'
        container.member = frob
        self.output_shape.type_name = 'structure'
        self.output_shape.members = {'Container': container}
        response = self.get_response()
        assert response is None

    def test_path_invalid(self):
        if False:
            return 10
        self.search_path = 'Container.Invalid'
        container = mock.Mock()
        container.type_name = 'invalid'
        self.output_shape.type_name = 'structure'
        self.output_shape.members = {'Container': container}
        with pytest.raises(NotImplementedError):
            self.get_response()

class TestRawHandler(BaseTestCase):

    def test_raw_handler_response(self):
        if False:
            i = 10
            return i + 15
        parent = mock.Mock()
        params = {}
        response = {'Id': 'foo'}
        handler = RawHandler(search_path=None)
        parsed_response = handler(parent, params, response)
        assert parsed_response == response

    def test_raw_handler_response_path(self):
        if False:
            for i in range(10):
                print('nop')
        parent = mock.Mock()
        params = {}
        frob = {'Id': 'foo'}
        response = {'Container': {'Frob': frob}}
        handler = RawHandler(search_path='Container.Frob')
        parsed_response = handler(parent, params, response)
        assert parsed_response == frob

class TestResourceHandler(BaseTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.identifier_path = ''
        self.factory = ResourceFactory(mock.Mock())
        self.resource_defs = {'Frob': {'shape': 'Frob', 'identifiers': [{'name': 'Id'}]}}
        self.service_model = mock.Mock()
        shape = mock.Mock()
        shape.members = {}
        self.service_model.shape_for.return_value = shape
        frobs = mock.Mock()
        frobs.type_name = 'list'
        container = mock.Mock()
        container.type_name = 'structure'
        container.members = {'Frobs': frobs}
        self.output_shape = mock.Mock()
        self.output_shape.type_name = 'structure'
        self.output_shape.members = {'Container': container}
        operation_model = mock.Mock()
        operation_model.output_shape = self.output_shape
        self.service_model.operation_model.return_value = operation_model
        self.parent = mock.Mock()
        self.parent.meta = ResourceMeta('test', client=mock.Mock())
        self.params = {}

    def get_resource(self, search_path, response):
        if False:
            print('Hello World!')
        request_resource_def = {'type': 'Frob', 'identifiers': [{'target': 'Id', 'source': 'response', 'path': self.identifier_path}]}
        resource_model = ResponseResource(request_resource_def, self.resource_defs)
        handler = ResourceHandler(search_path=search_path, factory=self.factory, resource_model=resource_model, service_context=ServiceContext(service_name='myservice', resource_json_definitions=self.resource_defs, service_model=self.service_model, service_waiter_model=None), operation_name='GetFrobs')
        return handler(self.parent, self.params, response)

    def test_create_resource_scalar(self):
        if False:
            print('Hello World!')
        self.identifier_path = 'Container.Id'
        search_path = 'Container'
        response = {'Container': {'Id': 'a-frob', 'OtherValue': 'other'}}
        resource = self.get_resource(search_path, response)
        assert isinstance(resource, ServiceResource)

    @mock.patch('boto3.resources.response.build_empty_response')
    def test_missing_data_scalar_builds_empty_response(self, build_mock):
        if False:
            for i in range(10):
                print('nop')
        self.identifier_path = 'Container.Id'
        search_path = 'Container'
        response = {'something': 'irrelevant'}
        resources = self.get_resource(search_path, response)
        assert build_mock.called
        assert resources == build_mock.return_value

    def test_create_resource_list(self):
        if False:
            return 10
        self.identifier_path = 'Container.Frobs[].Id'
        search_path = 'Container.Frobs[]'
        response = {'Container': {'Frobs': [{'Id': 'a-frob', 'OtherValue': 'other'}, {'Id': 'another-frob', 'OtherValue': 'foo'}]}}
        resources = self.get_resource(search_path, response)
        assert isinstance(resources, list)
        assert len(resources) == 2
        assert isinstance(resources[0], ServiceResource)

    def test_create_resource_list_no_search_path(self):
        if False:
            for i in range(10):
                print('nop')
        self.identifier_path = '[].Id'
        search_path = ''
        response = [{'Id': 'a-frob', 'OtherValue': 'other'}]
        resources = self.get_resource(search_path, response)
        assert isinstance(resources, list)
        assert len(resources) == 1
        assert isinstance(resources[0], ServiceResource)

    @mock.patch('boto3.resources.response.build_empty_response')
    def test_missing_data_list_builds_empty_response(self, build_mock):
        if False:
            print('Hello World!')
        self.identifier_path = 'Container.Frobs[].Id'
        search_path = 'Container.Frobs[]'
        response = {'something': 'irrelevant'}
        resources = self.get_resource(search_path, response)
        assert build_mock.called, 'build_empty_response was never called'
        assert resources == build_mock.return_value