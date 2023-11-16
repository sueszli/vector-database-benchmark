import sys
from typing import Any, List, Union
from unittest.mock import Mock
import pytest
from django.contrib.admin.views.decorators import staff_member_required
from django.test import Client, override_settings
from ninja import Body, Field, File, Form, NinjaAPI, Query, Schema, UploadedFile
from ninja.openapi.urls import get_openapi_urls
from ninja.pagination import PaginationBase, paginate
from ninja.renderers import JSONRenderer
api = NinjaAPI()

class Payload(Schema):
    i: int
    f: float

class TypeA(Schema):
    a: str

class TypeB(Schema):
    b: str

def to_camel(string: str) -> str:
    if False:
        while True:
            i = 10
    return ''.join((word.capitalize() for word in string.split('_')))

class Response(Schema):
    i: int
    f: float = Field(..., title='f title', description='f desc')

    class Config(Schema.Config):
        alias_generator = to_camel
        populate_by_name = True

@api.post('/test', response=Response)
def method(request, data: Payload):
    if False:
        i = 10
        return i + 15
    return data.dict()

@api.post('/test-alias', response=Response, by_alias=True)
def method_alias(request, data: Payload):
    if False:
        while True:
            i = 10
    return data.dict()

@api.post('/test_list', response=List[Response])
def method_list_response(request, data: List[Payload]):
    if False:
        i = 10
        return i + 15
    return []

@api.post('/test-body', response=Response)
def method_body(request, i: int=Body(...), f: float=Body(...)):
    if False:
        return 10
    return dict(i=i, f=f)

@api.post('/test-body-schema', response=Response)
def method_body_schema(request, data: Payload):
    if False:
        while True:
            i = 10
    return dict(i=data.i, f=data.f)

@api.get('/test-path/{int:i}/{f}', response=Response)
def method_path(request, i: int, f: float):
    if False:
        while True:
            i = 10
    return dict(i=i, f=f)

@api.post('/test-form', response=Response)
def method_form(request, data: Payload=Form(...)):
    if False:
        i = 10
        return i + 15
    return dict(i=data.i, f=data.f)

@api.post('/test-form-single', response=Response)
def method_form_single(request, data: float=Form(...)):
    if False:
        return 10
    return dict(i=int(data), f=data)

@api.post('/test-form-body', response=Response)
def method_form_body(request, i: int=Form(10), s: str=Body('10')):
    if False:
        while True:
            i = 10
    return dict(i=i, s=s)

@api.post('/test-form-file', response=Response)
def method_form_file(request, files: List[UploadedFile], data: Payload=Form(...)):
    if False:
        print('Hello World!')
    return dict(i=data.i, f=data.f)

@api.post('/test-body-file', response=Response)
def method_body_file(request, files: List[UploadedFile], body: Payload=Body(...)):
    if False:
        for i in range(10):
            print('nop')
    return dict(i=body.i, f=body.f)

@api.post('/test-union-type', response=Response)
def method_union_payload(request, data: Union[TypeA, TypeB]):
    if False:
        i = 10
        return i + 15
    return dict(i=data.i, f=data.f)

@api.post('/test-union-type-with-simple', response=Response)
def method_union_payload_and_simple(request, data: Union[int, TypeB]):
    if False:
        while True:
            i = 10
    return data.dict()
if sys.version_info >= (3, 10):

    @api.post('/test-new-union-type', response=Response)
    def method_new_union_payload(request, data: 'TypeA | TypeB'):
        if False:
            i = 10
            return i + 15
        return dict(i=data.i, f=data.f)

@api.post('/test-title-description/', tags=['a-tag'], summary='Best API Ever', response=Response)
def method_test_title_description(request, param1: int=Query(..., title='param 1 title'), param2: str=Query('A Default', description='param 2 desc'), file: UploadedFile=File(..., description='file param desc')):
    if False:
        return 10
    return dict(i=param1, f=param2)

@api.post('/test-deprecated-example-examples/')
def method_test_deprecated_example_examples(request, param1: int=Query(None, deprecated=True), param2: str=Query(..., example='Example Value'), param3: str=Query(..., max_length=5, examples={'normal': {'summary': 'A normal example', 'description': 'A **normal** string works correctly.', 'value': 'Foo'}, 'invalid': {'summary': 'Invalid data is rejected with an error', 'value': 'MoreThan5Length'}}), param4: int=Query(None, deprecated=True, include_in_schema=False)):
    if False:
        return 10
    return dict(i=param2, f=param3)

def test_schema_views(client: Client):
    if False:
        return 10
    assert client.get('/api/').status_code == 404
    assert client.get('/api/docs').status_code == 200
    assert client.get('/api/openapi.json').status_code == 200

def test_schema_views_no_INSTALLED_APPS(client: Client):
    if False:
        while True:
            i = 10
    'Making sure that cdn and included js works fine'
    from django.conf import settings
    INSTALLED_APPS = [i for i in settings.INSTALLED_APPS if i != 'ninja']

    @override_settings(INSTALLED_APPS=INSTALLED_APPS)
    def call_docs():
        if False:
            i = 10
            return i + 15
        assert client.get('/api/docs').status_code == 200
    call_docs()

@pytest.fixture(scope='session')
def schema():
    if False:
        i = 10
        return i + 15
    return api.get_openapi_schema()

def test_schema(schema):
    if False:
        while True:
            i = 10
    method = schema['paths']['/api/test']['post']
    assert method['requestBody'] == {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Payload'}}}, 'required': True}
    assert method['responses'] == {200: {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Response'}}}, 'description': 'OK'}}
    assert schema.schemas == {'Response': {'title': 'Response', 'type': 'object', 'properties': {'i': {'title': 'I', 'type': 'integer'}, 'f': {'description': 'f desc', 'title': 'f title', 'type': 'number'}}, 'required': ['i', 'f']}, 'Payload': {'title': 'Payload', 'type': 'object', 'properties': {'i': {'title': 'I', 'type': 'integer'}, 'f': {'title': 'F', 'type': 'number'}}, 'required': ['i', 'f']}, 'TypeA': {'properties': {'a': {'title': 'A', 'type': 'string'}}, 'required': ['a'], 'title': 'TypeA', 'type': 'object'}, 'TypeB': {'properties': {'b': {'title': 'B', 'type': 'string'}}, 'required': ['b'], 'title': 'TypeB', 'type': 'object'}}

def test_schema_alias(schema):
    if False:
        for i in range(10):
            print('nop')
    method = schema['paths']['/api/test-alias']['post']
    assert method['requestBody'] == {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Payload'}}}, 'required': True}
    assert method['responses'] == {200: {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Response'}}}, 'description': 'OK'}}
    '\n    assert schema.schemas == {\n        "Response": {\n            "title": "Response",\n            "type": "object",\n            "properties": {\n                "I": {"title": "I", "type": "integer"},\n                "F": {"title": "F", "type": "number"},\n            },\n            "required": ["i", "f"],\n        },\n        "Payload": {\n            "title": "Payload",\n            "type": "object",\n            "properties": {\n                "i": {"title": "I", "type": "integer"},\n                "f": {"title": "F", "type": "number"},\n            },\n            "required": ["i", "f"],\n        },\n    }\n    '

def test_schema_list(schema):
    if False:
        print('Hello World!')
    method_list = schema['paths']['/api/test_list']['post']
    assert method_list['requestBody'] == {'content': {'application/json': {'schema': {'items': {'$ref': '#/components/schemas/Payload'}, 'title': 'Data', 'type': 'array'}}}, 'required': True}
    assert method_list['responses'] == {200: {'content': {'application/json': {'schema': {'items': {'$ref': '#/components/schemas/Response'}, 'title': 'Response', 'type': 'array'}}}, 'description': 'OK'}}
    assert schema['components']['schemas'] == {'Payload': {'properties': {'f': {'title': 'F', 'type': 'number'}, 'i': {'title': 'I', 'type': 'integer'}}, 'required': ['i', 'f'], 'title': 'Payload', 'type': 'object'}, 'TypeA': {'properties': {'a': {'title': 'A', 'type': 'string'}}, 'required': ['a'], 'title': 'TypeA', 'type': 'object'}, 'TypeB': {'properties': {'b': {'title': 'B', 'type': 'string'}}, 'required': ['b'], 'title': 'TypeB', 'type': 'object'}, 'Response': {'properties': {'f': {'description': 'f desc', 'title': 'f title', 'type': 'number'}, 'i': {'title': 'I', 'type': 'integer'}}, 'required': ['i', 'f'], 'title': 'Response', 'type': 'object'}}

def test_schema_body(schema):
    if False:
        i = 10
        return i + 15
    method_list = schema['paths']['/api/test-body']['post']
    assert method_list['requestBody'] == {'content': {'application/json': {'schema': {'properties': {'f': {'title': 'F', 'type': 'number'}, 'i': {'title': 'I', 'type': 'integer'}}, 'required': ['i', 'f'], 'title': 'BodyParams', 'type': 'object'}}}, 'required': True}
    assert method_list['responses'] == {200: {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Response'}}}, 'description': 'OK'}}

def test_schema_body_schema(schema):
    if False:
        while True:
            i = 10
    method_list = schema['paths']['/api/test-body-schema']['post']
    assert method_list['requestBody'] == {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Payload'}}}, 'required': True}
    assert method_list['responses'] == {200: {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Response'}}}, 'description': 'OK'}}

def test_schema_path(schema):
    if False:
        return 10
    method_list = schema['paths']['/api/test-path/{i}/{f}']['get']
    assert 'requestBody' not in method_list
    assert method_list['parameters'] == [{'in': 'path', 'name': 'i', 'schema': {'title': 'I', 'type': 'integer'}, 'required': True}, {'in': 'path', 'name': 'f', 'schema': {'title': 'F', 'type': 'number'}, 'required': True}]
    assert method_list['responses'] == {200: {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Response'}}}, 'description': 'OK'}}

def test_schema_form(schema):
    if False:
        for i in range(10):
            print('nop')
    method_list = schema['paths']['/api/test-form']['post']
    assert method_list['requestBody'] == {'content': {'application/x-www-form-urlencoded': {'schema': {'title': 'FormParams', 'type': 'object', 'properties': {'i': {'title': 'I', 'type': 'integer'}, 'f': {'title': 'F', 'type': 'number'}}, 'required': ['i', 'f']}}}, 'required': True}
    assert method_list['responses'] == {200: {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Response'}}}}}

def test_schema_single(schema):
    if False:
        i = 10
        return i + 15
    method_list = schema['paths']['/api/test-form-single']['post']
    assert method_list['requestBody'] == {'content': {'application/x-www-form-urlencoded': {'schema': {'properties': {'data': {'title': 'Data', 'type': 'number'}}, 'required': ['data'], 'title': 'FormParams', 'type': 'object'}}}, 'required': True}
    assert method_list['responses'] == {200: {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Response'}}}}}

def test_schema_form_body(schema):
    if False:
        while True:
            i = 10
    method_list = schema['paths']['/api/test-form-body']['post']
    assert method_list['requestBody'] == {'content': {'multipart/form-data': {'schema': {'properties': {'i': {'default': 10, 'title': 'I', 'type': 'integer'}, 's': {'default': '10', 'title': 'S', 'type': 'string'}}, 'title': 'MultiPartBodyParams', 'type': 'object'}}}, 'required': True}
    assert method_list['responses'] == {200: {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Response'}}}}}

def test_schema_form_file(schema):
    if False:
        for i in range(10):
            print('nop')
    method_list = schema['paths']['/api/test-form-file']['post']
    assert method_list['requestBody'] == {'content': {'multipart/form-data': {'schema': {'properties': {'files': {'items': {'format': 'binary', 'type': 'string'}, 'title': 'Files', 'type': 'array'}, 'i': {'title': 'I', 'type': 'integer'}, 'f': {'title': 'F', 'type': 'number'}}, 'required': ['files', 'i', 'f'], 'title': 'MultiPartBodyParams', 'type': 'object'}}}, 'required': True}
    assert method_list['responses'] == {200: {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Response'}}}}}

def test_schema_body_file(schema):
    if False:
        return 10
    method_list = schema['paths']['/api/test-body-file']['post']
    assert method_list['requestBody'] == {'content': {'multipart/form-data': {'schema': {'properties': {'body': {'$ref': '#/components/schemas/Payload'}, 'files': {'items': {'format': 'binary', 'type': 'string'}, 'title': 'Files', 'type': 'array'}}, 'required': ['files', 'body'], 'title': 'MultiPartBodyParams', 'type': 'object'}}}, 'required': True}
    assert method_list['responses'] == {200: {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Response'}}}}}

def test_schema_title_description(schema):
    if False:
        i = 10
        return i + 15
    method_list = schema['paths']['/api/test-title-description/']['post']
    assert method_list['summary'] == 'Best API Ever'
    assert method_list['tags'] == ['a-tag']
    assert method_list['requestBody'] == {'content': {'multipart/form-data': {'schema': {'properties': {'file': {'description': 'file param desc', 'format': 'binary', 'title': 'File', 'type': 'string'}}, 'required': ['file'], 'title': 'FileParams', 'type': 'object'}}}, 'required': True}
    assert method_list['parameters'] == [{'in': 'query', 'name': 'param1', 'required': True, 'schema': {'title': 'param 1 title', 'type': 'integer'}}, {'in': 'query', 'name': 'param2', 'description': 'param 2 desc', 'required': False, 'schema': {'default': 'A Default', 'description': 'param 2 desc', 'title': 'Param2', 'type': 'string'}}]
    assert method_list['responses'] == {200: {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Response'}}}, 'description': 'OK'}}

def test_schema_deprecated_example_examples(schema):
    if False:
        i = 10
        return i + 15
    method_list = schema['paths']['/api/test-deprecated-example-examples/']['post']
    assert method_list['parameters'] == [{'deprecated': True, 'in': 'query', 'name': 'param1', 'required': False, 'schema': {'title': 'Param1', 'type': 'integer', 'deprecated': True}}, {'in': 'query', 'name': 'param2', 'required': True, 'schema': {'title': 'Param2', 'type': 'string', 'example': 'Example Value'}, 'example': 'Example Value'}, {'in': 'query', 'name': 'param3', 'required': True, 'schema': {'maxLength': 5, 'title': 'Param3', 'type': 'string', 'examples': {'invalid': {'summary': 'Invalid data is rejected with an error', 'value': 'MoreThan5Length'}, 'normal': {'description': 'A **normal** string works correctly.', 'summary': 'A normal example', 'value': 'Foo'}}}, 'examples': {'invalid': {'summary': 'Invalid data is rejected with an error', 'value': 'MoreThan5Length'}, 'normal': {'description': 'A **normal** string works correctly.', 'summary': 'A normal example', 'value': 'Foo'}}}]
    assert method_list['responses'] == {200: {'description': 'OK'}}

def test_union_payload_type(schema):
    if False:
        return 10
    method = schema['paths']['/api/test-union-type']['post']
    assert method['requestBody'] == {'content': {'application/json': {'schema': {'anyOf': [{'$ref': '#/components/schemas/TypeA'}, {'$ref': '#/components/schemas/TypeB'}], 'title': 'Data'}}}, 'required': True}

def test_union_payload_simple(schema):
    if False:
        for i in range(10):
            print('nop')
    method = schema['paths']['/api/test-union-type-with-simple']['post']
    print(method['requestBody'])
    assert method['requestBody'] == {'content': {'application/json': {'schema': {'title': 'Data', 'anyOf': [{'type': 'integer'}, {'$ref': '#/components/schemas/TypeB'}]}}}, 'required': True}

@pytest.mark.skipif(sys.version_info < (3, 10), reason='requires Python 3.10 or higher (PEP 604)')
def test_new_union_payload_type(schema):
    if False:
        while True:
            i = 10
    method = schema['paths']['/api/test-new-union-type']['post']
    assert method['requestBody'] == {'content': {'application/json': {'schema': {'anyOf': [{'$ref': '#/components/schemas/TypeA'}, {'$ref': '#/components/schemas/TypeB'}], 'title': 'Data'}}}, 'required': True}

def test_get_openapi_urls():
    if False:
        while True:
            i = 10
    api = NinjaAPI(openapi_url=None)
    paths = get_openapi_urls(api)
    assert len(paths) == 0
    api = NinjaAPI(docs_url=None)
    paths = get_openapi_urls(api)
    assert len(paths) == 1
    api = NinjaAPI(openapi_url='/path', docs_url='/path')
    with pytest.raises(AssertionError, match='Please use different urls for openapi_url and docs_url'):
        get_openapi_urls(api)

def test_unique_operation_ids():
    if False:
        return 10
    api = NinjaAPI()

    @api.get('/1')
    def same_name(request):
        if False:
            for i in range(10):
                print('nop')
        pass

    @api.get('/2')
    def same_name(request):
        if False:
            while True:
                i = 10
        pass
    match = 'operation_id "test_openapi_schema_same_name" is already used'
    with pytest.warns(UserWarning, match=match):
        api.get_openapi_schema()

def test_docs_decorator():
    if False:
        i = 10
        return i + 15
    api = NinjaAPI(docs_decorator=staff_member_required)
    paths = get_openapi_urls(api)
    assert len(paths) == 2
    for ptrn in paths:
        request = Mock(user=Mock(is_staff=True))
        result = ptrn.callback(request)
        assert result.status_code == 200
        request = Mock(user=Mock(is_staff=False))
        request.build_absolute_uri = lambda : 'http://example.com'
        result = ptrn.callback(request)
        assert result.status_code == 302

class TestRenderer(JSONRenderer):
    media_type = 'custom/type'

def test_renderer_media_type():
    if False:
        print('Hello World!')
    api = NinjaAPI(renderer=TestRenderer)

    @api.get('/1', response=TypeA)
    def same_name(request):
        if False:
            return 10
        pass
    schema = api.get_openapi_schema()
    method = schema['paths']['/api/1']['get']
    assert method['responses'] == {200: {'content': {'custom/type': {'schema': {'$ref': '#/components/schemas/TypeA'}}}, 'description': 'OK'}}

def test_all_paths_rendered():
    if False:
        while True:
            i = 10
    api = NinjaAPI(renderer=TestRenderer)

    @api.post('/1')
    def some_name_create(request):
        if False:
            return 10
        pass

    @api.get('/1')
    def some_name_list(request):
        if False:
            return 10
        pass

    @api.get('/1/{param}')
    def some_name_get_one(request, param: int):
        if False:
            return 10
        pass

    @api.delete('/1/{param}')
    def some_name_delete(request, param: int):
        if False:
            while True:
                i = 10
        pass
    schema = api.get_openapi_schema()
    expected_result = {'/api/1': ['post', 'get'], '/api/1/{param}': ['get', 'delete']}
    result = {p: list(schema['paths'][p].keys()) for p in schema['paths'].keys()}
    assert expected_result == result

def test_all_paths_typed_params_rendered():
    if False:
        return 10
    api = NinjaAPI(renderer=TestRenderer)

    @api.post('/1')
    def some_name_create(request):
        if False:
            for i in range(10):
                print('nop')
        pass

    @api.get('/1')
    def some_name_list(request):
        if False:
            for i in range(10):
                print('nop')
        pass

    @api.get('/1/{int:param}')
    def some_name_get_one(request, param: int):
        if False:
            for i in range(10):
                print('nop')
        pass

    @api.delete('/1/{str:param}')
    def some_name_delete(request, param: str):
        if False:
            i = 10
            return i + 15
        pass
    schema = api.get_openapi_schema()
    expected_result = {'/api/1': ['post', 'get'], '/api/1/{param}': ['get', 'delete']}
    result = {p: list(schema['paths'][p].keys()) for p in schema['paths'].keys()}
    assert expected_result == result

def test_no_default_for_custom_items_attribute():
    if False:
        for i in range(10):
            print('nop')
    api = NinjaAPI(renderer=TestRenderer)

    class EmployeeOut(Schema):
        id: int
        first_name: str
        last_name: str

    class CustomPagination(PaginationBase):

        class Output(Schema):
            data: List[Any]
            detail: str
            total: int
        items_attribute: str = 'data'

        def paginate_queryset(self, queryset, pagination, **params):
            if False:
                print('Hello World!')
            pass

    @api.get('/employees', auth=['OAuth'], response=List[EmployeeOut])
    @paginate(CustomPagination)
    def get_employees(request):
        if False:
            return 10
        pass
    schema = api.get_openapi_schema()
    paged_employee_out = schema['components']['schemas']['PagedEmployeeOut']
    assert 'default' not in paged_employee_out['properties']['data']