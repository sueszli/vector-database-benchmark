from typing import List, Optional
import pytest
from fastapi import FastAPI
from fastapi.exceptions import ResponseValidationError
from fastapi.testclient import TestClient
from pydantic.dataclasses import dataclass
app = FastAPI()

@dataclass
class Item:
    name: str
    price: Optional[float] = None
    owner_ids: Optional[List[int]] = None

@app.get('/items/invalid', response_model=Item)
def get_invalid():
    if False:
        print('Hello World!')
    return {'name': 'invalid', 'price': 'foo'}

@app.get('/items/innerinvalid', response_model=Item)
def get_innerinvalid():
    if False:
        for i in range(10):
            print('nop')
    return {'name': 'double invalid', 'price': 'foo', 'owner_ids': ['foo', 'bar']}

@app.get('/items/invalidlist', response_model=List[Item])
def get_invalidlist():
    if False:
        return 10
    return [{'name': 'foo'}, {'name': 'bar', 'price': 'bar'}, {'name': 'baz', 'price': 'baz'}]
client = TestClient(app)

def test_invalid():
    if False:
        while True:
            i = 10
    with pytest.raises(ResponseValidationError):
        client.get('/items/invalid')

def test_double_invalid():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ResponseValidationError):
        client.get('/items/innerinvalid')

def test_invalid_list():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ResponseValidationError):
        client.get('/items/invalidlist')