from connexion.mock import MockResolver
from connexion.operations import OpenAPIOperation

def test_mock_resolver_default():
    if False:
        i = 10
        return i + 15
    resolver = MockResolver(mock_all=True)
    responses = {'default': {'content': {'application/json': {'examples': {'super_cool_example': {'value': {'foo': 'bar'}}}}}}}
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'responses': responses}, resolver=resolver)
    assert operation.operation_id == 'mock-1'
    (response, status_code) = resolver.mock_operation(operation)
    assert status_code == 200
    assert response == {'foo': 'bar'}

def test_mock_resolver_numeric():
    if False:
        print('Hello World!')
    resolver = MockResolver(mock_all=True)
    responses = {'200': {'content': {'application/json': {'examples': {'super_cool_example': {'value': {'foo': 'bar'}}}}}}}
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'responses': responses}, resolver=resolver)
    assert operation.operation_id == 'mock-1'
    (response, status_code) = resolver.mock_operation(operation)
    assert status_code == 200
    assert response == {'foo': 'bar'}

def test_mock_resolver_inline_schema_example():
    if False:
        i = 10
        return i + 15
    resolver = MockResolver(mock_all=True)
    responses = {'default': {'content': {'application/json': {'schema': {'type': 'object', 'properties': {'foo': {'schema': {'type': 'string'}}}}, 'example': {'foo': 'bar'}}}}}
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'responses': responses}, resolver=resolver)
    assert operation.operation_id == 'mock-1'
    (response, status_code) = resolver.mock_operation(operation)
    assert status_code == 200
    assert response == {'foo': 'bar'}

def test_mock_resolver_no_examples():
    if False:
        return 10
    resolver = MockResolver(mock_all=True)
    responses = {'418': {}}
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'responses': responses}, resolver=resolver)
    assert operation.operation_id == 'mock-1'
    (response, status_code) = resolver.mock_operation(operation)
    assert status_code == 418
    assert response == 'No example response was defined.'

def test_mock_resolver_notimplemented():
    if False:
        print('Hello World!')
    resolver = MockResolver(mock_all=False)
    responses = {'418': {}}
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'operationId': 'fakeapi.hello.get'}, resolver=resolver)
    assert operation.operation_id == 'fakeapi.hello.get'
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'operationId': 'fakeapi.hello.nonexistent_function', 'responses': responses}, resolver=resolver)
    assert operation._resolution.function() == ('No example response was defined.', 418)