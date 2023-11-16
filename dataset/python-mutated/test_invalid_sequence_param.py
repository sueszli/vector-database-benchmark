from typing import Dict, List, Optional, Tuple
import pytest
from fastapi import FastAPI, Query
from pydantic import BaseModel

def test_invalid_sequence():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(AssertionError):
        app = FastAPI()

        class Item(BaseModel):
            title: str

        @app.get('/items/')
        def read_items(q: List[Item]=Query(default=None)):
            if False:
                return 10
            pass

def test_invalid_tuple():
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError):
        app = FastAPI()

        class Item(BaseModel):
            title: str

        @app.get('/items/')
        def read_items(q: Tuple[Item, Item]=Query(default=None)):
            if False:
                i = 10
                return i + 15
            pass

def test_invalid_dict():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(AssertionError):
        app = FastAPI()

        class Item(BaseModel):
            title: str

        @app.get('/items/')
        def read_items(q: Dict[str, Item]=Query(default=None)):
            if False:
                for i in range(10):
                    print('nop')
            pass

def test_invalid_simple_dict():
    if False:
        i = 10
        return i + 15
    with pytest.raises(AssertionError):
        app = FastAPI()

        class Item(BaseModel):
            title: str

        @app.get('/items/')
        def read_items(q: Optional[dict]=Query(default=None)):
            if False:
                print('Hello World!')
            pass