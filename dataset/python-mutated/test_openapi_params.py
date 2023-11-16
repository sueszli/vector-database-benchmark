from ninja import NinjaAPI
api = NinjaAPI()

@api.get('/operation1', operation_id='my_id')
def operation_1(request):
    if False:
        for i in range(10):
            print('nop')
    '\n    This will be in description\n    '
    return {'docstrings': True}

@api.get('/operation2', description='description from argument', deprecated=True)
def operation2(request):
    if False:
        i = 10
        return i + 15
    return {'description': True, 'deprecated': True}

@api.get('/operation3', summary='Summary from argument', description='description arg')
def operation3(request):
    if False:
        print('Hello World!')
    'This one also has docstring description'
    return {'summary': True, 'description': 'multiple'}

@api.get('/operation4', tags=['tag1', 'tag2'])
def operation4(request):
    if False:
        return 10
    return {'tags': True}

@api.get('/operation5', openapi_extra={'requestBody': {'content': {'application/json': {'schema': {'required': ['email'], 'type': 'object', 'properties': {'name': {'type': 'string'}, 'phone': {'type': 'number'}, 'email': {'type': 'string'}}}}}, 'required': True}})
def operation5(request):
    if False:
        for i in range(10):
            print('nop')
    return {'openapi_extra': True}

@api.get('/not-included', include_in_schema=False)
def not_included(request):
    if False:
        i = 10
        return i + 15
    return True
schema = api.get_openapi_schema()

def test_schema():
    if False:
        return 10
    op1 = schema['paths']['/api/operation1']['get']
    op2 = schema['paths']['/api/operation2']['get']
    op3 = schema['paths']['/api/operation3']['get']
    op4 = schema['paths']['/api/operation4']['get']
    op5 = schema['paths']['/api/operation5']['get']
    assert op1['operationId'] == 'my_id'
    assert op2['operationId'] == 'test_openapi_params_operation2'
    assert op3['operationId'] == 'test_openapi_params_operation3'
    assert op4['operationId'] == 'test_openapi_params_operation4'
    assert op5['operationId'] == 'test_openapi_params_operation5'
    assert op1['summary'] == 'Operation 1'
    assert op2['summary'] == 'Operation2'
    assert op3['summary'] == 'Summary from argument'
    assert op4['summary'] == 'Operation4'
    assert op5['summary'] == 'Operation5'
    assert op1['description'] == 'This will be in description'
    assert op2['description'] == 'description from argument'
    assert op2['deprecated'] is True
    assert op3['description'] == 'description arg'
    assert op4['tags'] == ['tag1', 'tag2']
    assert op5['requestBody'] == {'content': {'application/json': {'schema': {'required': ['email'], 'type': 'object', 'properties': {'name': {'type': 'string'}, 'phone': {'type': 'number'}, 'email': {'type': 'string'}}}}}, 'required': True}

def test_not_included():
    if False:
        while True:
            i = 10
    assert list(schema['paths'].keys()) == ['/api/operation1', '/api/operation2', '/api/operation3', '/api/operation4', '/api/operation5']