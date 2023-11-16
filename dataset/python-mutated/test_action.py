import pytest
from boto3.resources.action import BatchAction, ServiceAction, WaiterAction
from boto3.resources.base import ResourceMeta
from boto3.resources.model import Action, Waiter
from boto3.utils import ServiceContext
from tests import BaseTestCase, mock

class TestServiceActionCall(BaseTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.action_def = {'request': {'operation': 'GetFrobs', 'params': []}}

    @property
    def action(self):
        if False:
            print('Hello World!')
        return Action('test', self.action_def, {})

    @mock.patch('boto3.resources.action.create_request_parameters', return_value={})
    def test_service_action_creates_params(self, params_mock):
        if False:
            return 10
        resource = mock.Mock()
        resource.meta = ResourceMeta('test', client=mock.Mock())
        action = ServiceAction(self.action)
        action(resource, foo=1)
        assert params_mock.called

    @mock.patch('boto3.resources.action.create_request_parameters', return_value={'bar': 'baz'})
    def test_service_action_calls_operation(self, params_mock):
        if False:
            i = 10
            return i + 15
        resource = mock.Mock()
        resource.meta = ResourceMeta('test', client=mock.Mock())
        operation = resource.meta.client.get_frobs
        operation.return_value = 'response'
        action = ServiceAction(self.action)
        response = action(resource, foo=1)
        operation.assert_called_with(foo=1, bar='baz')
        assert response == 'response'

    @mock.patch('boto3.resources.action.create_request_parameters', return_value={})
    @mock.patch('boto3.resources.action.RawHandler')
    def test_service_action_calls_raw_handler(self, handler_mock, params_mock):
        if False:
            print('Hello World!')
        resource = mock.Mock()
        resource.meta = ResourceMeta('test', client=mock.Mock())
        operation = resource.meta.client.get_frobs
        operation.return_value = 'response'
        action = ServiceAction(self.action)
        handler_mock.return_value.return_value = 'response'
        action(resource)
        handler_mock.assert_called_with(None)
        handler_mock.return_value.assert_called_with(resource, {}, 'response')

    @mock.patch('boto3.resources.action.create_request_parameters', return_value={})
    @mock.patch('boto3.resources.action.ResourceHandler')
    def test_service_action_calls_resource_handler(self, handler_mock, params_mock):
        if False:
            print('Hello World!')
        self.action_def['resource'] = {'type': 'Frob', 'path': 'Container'}
        resource = mock.Mock()
        resource.meta = ResourceMeta('test', client=mock.Mock())
        operation = resource.meta.client.get_frobs
        operation.return_value = 'response'
        factory = mock.Mock()
        resource_defs = {}
        service_model = mock.Mock()
        action_model = self.action
        service_context = ServiceContext(service_name='test', service_model=service_model, resource_json_definitions=resource_defs, service_waiter_model=None)
        action = ServiceAction(action_model=action_model, factory=factory, service_context=service_context)
        handler_mock.return_value.return_value = 'response'
        action(resource)
        handler_mock.assert_called_with(search_path='Container', factory=factory, resource_model=action_model.resource, service_context=service_context, operation_name='GetFrobs')

    def test_service_action_call_positional_argument(self):
        if False:
            return 10

        def _api_call(*args, **kwargs):
            if False:
                while True:
                    i = 10
            if args:
                raise TypeError('%s() only accepts keyword arguments.' % 'get_frobs')
        resource = mock.Mock()
        resource.meta = ResourceMeta('test', client=mock.Mock())
        resource.meta.client.get_frobs = _api_call
        action = ServiceAction(self.action)
        with pytest.raises(TypeError):
            action(resource, 'item1')

class TestWaiterActionCall(BaseTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.waiter_resource_name = 'wait_until_exists'
        self.waiter_def = {'waiterName': 'FrobExists', 'params': [{'target': 'Frob', 'sourceType': 'identifier', 'source': 'Name'}]}

    @property
    def waiter(self):
        if False:
            return 10
        return Waiter('test', self.waiter_def)

    @mock.patch('boto3.resources.action.create_request_parameters', return_value={})
    def test_service_waiter_creates_params(self, params_mock):
        if False:
            return 10
        resource = mock.Mock()
        resource.meta = ResourceMeta('test', client=mock.Mock())
        action = WaiterAction(self.waiter, self.waiter_resource_name)
        action(resource, foo=1)
        assert params_mock.called

    @mock.patch('boto3.resources.action.create_request_parameters', return_value={'bar': 'baz'})
    def test_service_action_calls_operation(self, params_mock):
        if False:
            for i in range(10):
                print('nop')
        resource = mock.Mock()
        resource.meta = ResourceMeta('test', client=mock.Mock())
        get_waiter = resource.meta.client.get_waiter
        mock_waiter = mock.Mock()
        get_waiter.return_value = mock_waiter
        action = WaiterAction(self.waiter, self.waiter_resource_name)
        action(resource, foo=1)
        get_waiter.assert_called_with('frob_exists')
        mock_waiter.wait.assert_called_with(foo=1, bar='baz')

class TestBatchActionCall(BaseTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.action_def = {'request': {'operation': 'GetFrobs', 'params': []}}

    @property
    def model(self):
        if False:
            return 10
        return Action('test', self.action_def, {})

    def test_batch_action_gets_pages_from_collection(self):
        if False:
            for i in range(10):
                print('nop')
        collection = mock.Mock()
        collection.pages.return_value = []
        action = BatchAction(self.model)
        action(collection)
        collection.pages.assert_called_with()

    def test_batch_action_creates_parameters_from_items(self):
        if False:
            while True:
                i = 10
        self.action_def['request']['params'] = [{'target': 'Bucket', 'source': 'data', 'path': 'BucketName'}, {'target': 'Delete.Objects[].Key', 'source': 'data', 'path': 'Key'}]
        client = mock.Mock()
        item1 = mock.Mock()
        item1.meta = ResourceMeta('test', client=client, data={'BucketName': 'bucket', 'Key': 'item1'})
        item2 = mock.Mock()
        item2.meta = ResourceMeta('test', client=client, data={'BucketName': 'bucket', 'Key': 'item2'})
        collection = mock.Mock()
        collection.pages.return_value = [[item1, item2]]
        action = BatchAction(self.model)
        action(collection)
        client.get_frobs.assert_called_with(Bucket='bucket', Delete={'Objects': [{'Key': 'item1'}, {'Key': 'item2'}]})

    @mock.patch('boto3.resources.action.create_request_parameters', return_value={})
    def test_batch_action_skips_operation(self, crp_mock):
        if False:
            while True:
                i = 10
        client = mock.Mock()
        item = mock.Mock()
        item.meta = ResourceMeta('test', client=client)
        collection = mock.Mock()
        collection.pages.return_value = [[item]]
        model = self.model
        action = BatchAction(model)
        action(collection)
        crp_mock.assert_called_with(item, model.request, params={}, index=0)
        client.get_frobs.assert_not_called()

    @mock.patch('boto3.resources.action.create_request_parameters')
    def test_batch_action_calls_operation(self, crp_mock):
        if False:
            for i in range(10):
                print('nop')

        def side_effect(resource, model, params=None, index=None):
            if False:
                i = 10
                return i + 15
            params['foo'] = 'bar'
        crp_mock.side_effect = side_effect
        client = mock.Mock()
        item = mock.Mock()
        item.meta = ResourceMeta('test', client=client)
        collection = mock.Mock()
        collection.pages.return_value = [[item]]
        model = self.model
        action = BatchAction(model)
        action(collection)
        crp_mock.assert_called_with(item, model.request, params={'foo': 'bar'}, index=0)
        client.get_frobs.assert_called_with(foo='bar')

    @mock.patch('boto3.resources.action.create_request_parameters')
    def test_batch_action_with_positional_argument(self, crp_mock):
        if False:
            for i in range(10):
                print('nop')

        def side_effect(resource, model, params=None, index=None):
            if False:
                i = 10
                return i + 15
            params['foo'] = 'bar'

        def _api_call(*args, **kwargs):
            if False:
                print('Hello World!')
            if args:
                raise TypeError('%s() only accepts keyword arguments.' % 'get_frobs')
        crp_mock.side_effect = side_effect
        client = mock.Mock()
        client.get_frobs = _api_call
        item = mock.Mock()
        item.meta = ResourceMeta('test', client=client)
        collection = mock.Mock()
        collection.pages.return_value = [[item]]
        model = self.model
        action = BatchAction(model)
        with pytest.raises(TypeError):
            action(collection, 'item1')