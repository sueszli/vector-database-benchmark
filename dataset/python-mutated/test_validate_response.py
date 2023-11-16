from typing import List, Optional, Union
import pytest
from fastapi import FastAPI
from fastapi.exceptions import ResponseValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel
app = FastAPI()

class Item(BaseModel):
    name: str
    price: Optional[float] = None
    owner_ids: Optional[List[int]] = None

@app.get('/items/invalid', response_model=Item)
def get_invalid():
    if False:
        return 10
    return {'name': 'invalid', 'price': 'foo'}

@app.get('/items/invalidnone', response_model=Item)
def get_invalid_none():
    if False:
        while True:
            i = 10
    return None

@app.get('/items/validnone', response_model=Union[Item, None])
def get_valid_none(send_none: bool=False):
    if False:
        i = 10
        return i + 15
    if send_none:
        return None
    else:
        return {'name': 'invalid', 'price': 3.2}

@app.get('/items/innerinvalid', response_model=Item)
def get_innerinvalid():
    if False:
        return 10
    return {'name': 'double invalid', 'price': 'foo', 'owner_ids': ['foo', 'bar']}

@app.get('/items/invalidlist', response_model=List[Item])
def get_invalidlist():
    if False:
        while True:
            i = 10
    return [{'name': 'foo'}, {'name': 'bar', 'price': 'bar'}, {'name': 'baz', 'price': 'baz'}]
client = TestClient(app)

def test_invalid():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ResponseValidationError):
        client.get('/items/invalid')

def test_invalid_none():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ResponseValidationError):
        client.get('/items/invalidnone')

def test_valid_none_data():
    if False:
        while True:
            i = 10
    response = client.get('/items/validnone')
    data = response.json()
    assert response.status_code == 200
    assert data == {'name': 'invalid', 'price': 3.2, 'owner_ids': None}

def test_valid_none_none():
    if False:
        print('Hello World!')
    response = client.get('/items/validnone', params={'send_none': 'true'})
    data = response.json()
    assert response.status_code == 200
    assert data is None

def test_double_invalid():
    if False:
        while True:
            i = 10
    with pytest.raises(ResponseValidationError):
        client.get('/items/innerinvalid')

def test_invalid_list():
    if False:
        print('Hello World!')
    with pytest.raises(ResponseValidationError):
        client.get('/items/invalidlist')