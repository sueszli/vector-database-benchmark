from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

class Model1(BaseModel):
    foo: str
    bar: str

class Model2(BaseModel):
    ref: Model1
    baz: str

class Model3(BaseModel):
    name: str
    age: int
    ref2: Model2
app = FastAPI()

@app.get('/simple_include', response_model=Model2, response_model_include={'baz': ..., 'ref': {'foo'}})
def simple_include():
    if False:
        return 10
    return Model2(ref=Model1(foo='simple_include model foo', bar='simple_include model bar'), baz='simple_include model2 baz')

@app.get('/simple_include_dict', response_model=Model2, response_model_include={'baz': ..., 'ref': {'foo'}})
def simple_include_dict():
    if False:
        i = 10
        return i + 15
    return {'ref': {'foo': 'simple_include_dict model foo', 'bar': 'simple_include_dict model bar'}, 'baz': 'simple_include_dict model2 baz'}

@app.get('/simple_exclude', response_model=Model2, response_model_exclude={'ref': {'bar'}})
def simple_exclude():
    if False:
        while True:
            i = 10
    return Model2(ref=Model1(foo='simple_exclude model foo', bar='simple_exclude model bar'), baz='simple_exclude model2 baz')

@app.get('/simple_exclude_dict', response_model=Model2, response_model_exclude={'ref': {'bar'}})
def simple_exclude_dict():
    if False:
        return 10
    return {'ref': {'foo': 'simple_exclude_dict model foo', 'bar': 'simple_exclude_dict model bar'}, 'baz': 'simple_exclude_dict model2 baz'}

@app.get('/mixed', response_model=Model3, response_model_include={'ref2', 'name'}, response_model_exclude={'ref2': {'baz'}})
def mixed():
    if False:
        print('Hello World!')
    return Model3(name='mixed model3 name', age=3, ref2=Model2(ref=Model1(foo='mixed model foo', bar='mixed model bar'), baz='mixed model2 baz'))

@app.get('/mixed_dict', response_model=Model3, response_model_include={'ref2', 'name'}, response_model_exclude={'ref2': {'baz'}})
def mixed_dict():
    if False:
        for i in range(10):
            print('nop')
    return {'name': 'mixed_dict model3 name', 'age': 3, 'ref2': {'ref': {'foo': 'mixed_dict model foo', 'bar': 'mixed_dict model bar'}, 'baz': 'mixed_dict model2 baz'}}
client = TestClient(app)

def test_nested_include_simple():
    if False:
        while True:
            i = 10
    response = client.get('/simple_include')
    assert response.status_code == 200, response.text
    assert response.json() == {'baz': 'simple_include model2 baz', 'ref': {'foo': 'simple_include model foo'}}

def test_nested_include_simple_dict():
    if False:
        while True:
            i = 10
    response = client.get('/simple_include_dict')
    assert response.status_code == 200, response.text
    assert response.json() == {'baz': 'simple_include_dict model2 baz', 'ref': {'foo': 'simple_include_dict model foo'}}

def test_nested_exclude_simple():
    if False:
        print('Hello World!')
    response = client.get('/simple_exclude')
    assert response.status_code == 200, response.text
    assert response.json() == {'baz': 'simple_exclude model2 baz', 'ref': {'foo': 'simple_exclude model foo'}}

def test_nested_exclude_simple_dict():
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/simple_exclude_dict')
    assert response.status_code == 200, response.text
    assert response.json() == {'baz': 'simple_exclude_dict model2 baz', 'ref': {'foo': 'simple_exclude_dict model foo'}}

def test_nested_include_mixed():
    if False:
        i = 10
        return i + 15
    response = client.get('/mixed')
    assert response.status_code == 200, response.text
    assert response.json() == {'name': 'mixed model3 name', 'ref2': {'ref': {'foo': 'mixed model foo', 'bar': 'mixed model bar'}}}

def test_nested_include_mixed_dict():
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/mixed_dict')
    assert response.status_code == 200, response.text
    assert response.json() == {'name': 'mixed_dict model3 name', 'ref2': {'ref': {'foo': 'mixed_dict model foo', 'bar': 'mixed_dict model bar'}}}