from typing import Dict, List, Tuple
import pytest
from fastapi import FastAPI
from pydantic import BaseModel

def test_invalid_sequence():
    if False:
        return 10
    with pytest.raises(AssertionError):
        app = FastAPI()

        class Item(BaseModel):
            title: str

        @app.get('/items/{id}')
        def read_items(id: List[Item]):
            if False:
                i = 10
                return i + 15
            pass

def test_invalid_tuple():
    if False:
        i = 10
        return i + 15
    with pytest.raises(AssertionError):
        app = FastAPI()

        class Item(BaseModel):
            title: str

        @app.get('/items/{id}')
        def read_items(id: Tuple[Item, Item]):
            if False:
                for i in range(10):
                    print('nop')
            pass

def test_invalid_dict():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(AssertionError):
        app = FastAPI()

        class Item(BaseModel):
            title: str

        @app.get('/items/{id}')
        def read_items(id: Dict[str, Item]):
            if False:
                i = 10
                return i + 15
            pass

def test_invalid_simple_list():
    if False:
        return 10
    with pytest.raises(AssertionError):
        app = FastAPI()

        @app.get('/items/{id}')
        def read_items(id: list):
            if False:
                print('Hello World!')
            pass

def test_invalid_simple_tuple():
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError):
        app = FastAPI()

        @app.get('/items/{id}')
        def read_items(id: tuple):
            if False:
                return 10
            pass

def test_invalid_simple_set():
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError):
        app = FastAPI()

        @app.get('/items/{id}')
        def read_items(id: set):
            if False:
                while True:
                    i = 10
            pass

def test_invalid_simple_dict():
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError):
        app = FastAPI()

        @app.get('/items/{id}')
        def read_items(id: dict):
            if False:
                print('Hello World!')
            pass