from connexion.operations import OpenAPIOperation
from connexion.resolver import RelativeResolver, Resolver, RestyResolver
COMPONENTS = {'parameters': {'myparam': {'in': 'path', 'schema': {'type': 'integer'}}}}

def test_standard_resolve_x_router_controller():
    if False:
        i = 10
        return i + 15
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'x-openapi-router-controller': 'fakeapi.hello', 'operationId': 'post_greeting'}, components=COMPONENTS, resolver=Resolver())
    assert operation.operation_id == 'fakeapi.hello.post_greeting'

def test_relative_resolve_x_router_controller():
    if False:
        i = 10
        return i + 15
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'x-openapi-router-controller': 'fakeapi.hello', 'operationId': 'post_greeting'}, components=COMPONENTS, resolver=RelativeResolver('root_path'))
    assert operation.operation_id == 'fakeapi.hello.post_greeting'

def test_relative_resolve_operation_id():
    if False:
        i = 10
        return i + 15
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'operationId': 'hello.post_greeting'}, components=COMPONENTS, resolver=RelativeResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.hello.post_greeting'

def test_relative_resolve_operation_id_with_module():
    if False:
        i = 10
        return i + 15
    import fakeapi
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'operationId': 'hello.post_greeting'}, components=COMPONENTS, resolver=RelativeResolver(fakeapi))
    assert operation.operation_id == 'fakeapi.hello.post_greeting'

def test_resty_resolve_operation_id():
    if False:
        for i in range(10):
            print('nop')
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'operationId': 'fakeapi.hello.post_greeting'}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.hello.post_greeting'

def test_resty_resolve_x_router_controller_with_operation_id():
    if False:
        print('Hello World!')
    operation = OpenAPIOperation(method='GET', path='endpoint', path_parameters=[], operation={'x-openapi-router-controller': 'fakeapi.hello', 'operationId': 'post_greeting'}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.hello.post_greeting'

def test_resty_resolve_x_router_controller_without_operation_id():
    if False:
        return 10
    operation = OpenAPIOperation(method='GET', path='/hello/{id}', path_parameters=[], operation={'x-openapi-router-controller': 'fakeapi.hello'}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.hello.get'

def test_resty_resolve_with_default_module_name():
    if False:
        for i in range(10):
            print('nop')
    operation = OpenAPIOperation(method='GET', path='/hello/{id}', path_parameters=[], operation={}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.hello.get'

def test_resty_resolve_with_default_module_name():
    if False:
        i = 10
        return i + 15
    operation = OpenAPIOperation(method='GET', path='/hello/{id}/world', path_parameters=[], operation={}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.hello.world.search'

def test_resty_resolve_with_default_module_name_lowercase_verb():
    if False:
        print('Hello World!')
    operation = OpenAPIOperation(method='get', path='/hello/{id}', path_parameters=[], operation={}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.hello.get'

def test_resty_resolve_with_default_module_name_lowercase_verb_nested():
    if False:
        for i in range(10):
            print('nop')
    operation = OpenAPIOperation(method='get', path='/hello/world/{id}', path_parameters=[], operation={}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.hello.world.get'

def test_resty_resolve_with_default_module_name_will_translate_dashes_in_resource_name():
    if False:
        i = 10
        return i + 15
    operation = OpenAPIOperation(method='GET', path='/foo-bar', path_parameters=[], operation={}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.foo_bar.search'

def test_resty_resolve_with_default_module_name_can_resolve_api_root():
    if False:
        return 10
    operation = OpenAPIOperation(method='GET', path='/', path_parameters=[], operation={}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.get'

def test_resty_resolve_with_default_module_name_will_resolve_resource_root_get_as_search():
    if False:
        while True:
            i = 10
    operation = OpenAPIOperation(method='GET', path='/hello', path_parameters=[], operation={}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.hello.search'

def test_resty_resolve_with_default_module_name_and_x_router_controller_will_resolve_resource_root_get_as_search():
    if False:
        while True:
            i = 10
    operation = OpenAPIOperation(method='GET', path='/hello', path_parameters=[], operation={'x-openapi-router-controller': 'fakeapi.hello'}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.hello.search'

def test_resty_resolve_with_default_module_name_will_resolve_resource_root_as_configured():
    if False:
        for i in range(10):
            print('nop')
    operation = OpenAPIOperation(method='GET', path='/hello', path_parameters=[], operation={}, components=COMPONENTS, resolver=RestyResolver('fakeapi', collection_endpoint_name='api_list'))
    assert operation.operation_id == 'fakeapi.hello.api_list'

def test_resty_resolve_with_default_module_name_will_resolve_resource_root_post_as_post():
    if False:
        for i in range(10):
            print('nop')
    operation = OpenAPIOperation(method='POST', path='/hello', path_parameters=[], operation={}, components=COMPONENTS, resolver=RestyResolver('fakeapi'))
    assert operation.operation_id == 'fakeapi.hello.post'