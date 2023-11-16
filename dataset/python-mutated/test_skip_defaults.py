from typing import Optional
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
app = FastAPI()

class SubModel(BaseModel):
    a: Optional[str] = 'foo'

class Model(BaseModel):
    x: Optional[int] = None
    sub: SubModel

class ModelSubclass(Model):
    y: int
    z: int = 0
    w: Optional[int] = None

class ModelDefaults(BaseModel):
    w: Optional[str] = None
    x: Optional[str] = None
    y: str = 'y'
    z: str = 'z'

@app.get('/', response_model=Model, response_model_exclude_unset=True)
def get_root() -> ModelSubclass:
    if False:
        for i in range(10):
            print('nop')
    return ModelSubclass(sub={}, y=1, z=0)

@app.get('/exclude_unset', response_model=ModelDefaults, response_model_exclude_unset=True)
def get_exclude_unset() -> ModelDefaults:
    if False:
        print('Hello World!')
    return ModelDefaults(x=None, y='y')

@app.get('/exclude_defaults', response_model=ModelDefaults, response_model_exclude_defaults=True)
def get_exclude_defaults() -> ModelDefaults:
    if False:
        print('Hello World!')
    return ModelDefaults(x=None, y='y')

@app.get('/exclude_none', response_model=ModelDefaults, response_model_exclude_none=True)
def get_exclude_none() -> ModelDefaults:
    if False:
        for i in range(10):
            print('nop')
    return ModelDefaults(x=None, y='y')

@app.get('/exclude_unset_none', response_model=ModelDefaults, response_model_exclude_unset=True, response_model_exclude_none=True)
def get_exclude_unset_none() -> ModelDefaults:
    if False:
        print('Hello World!')
    return ModelDefaults(x=None, y='y')
client = TestClient(app)

def test_return_defaults():
    if False:
        return 10
    response = client.get('/')
    assert response.json() == {'sub': {}}

def test_return_exclude_unset():
    if False:
        while True:
            i = 10
    response = client.get('/exclude_unset')
    assert response.json() == {'x': None, 'y': 'y'}

def test_return_exclude_defaults():
    if False:
        print('Hello World!')
    response = client.get('/exclude_defaults')
    assert response.json() == {}

def test_return_exclude_none():
    if False:
        i = 10
        return i + 15
    response = client.get('/exclude_none')
    assert response.json() == {'y': 'y', 'z': 'z'}

def test_return_exclude_unset_none():
    if False:
        return 10
    response = client.get('/exclude_unset_none')
    assert response.json() == {'y': 'y'}