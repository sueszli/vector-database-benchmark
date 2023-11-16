from typing import List
import pytest
from fastapi import FastAPI
from fastapi.exceptions import FastAPIError

class NonPydanticModel:
    pass

def test_invalid_response_model_raises():
    if False:
        i = 10
        return i + 15
    with pytest.raises(FastAPIError):
        app = FastAPI()

        @app.get('/', response_model=NonPydanticModel)
        def read_root():
            if False:
                while True:
                    i = 10
            pass

def test_invalid_response_model_sub_type_raises():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(FastAPIError):
        app = FastAPI()

        @app.get('/', response_model=List[NonPydanticModel])
        def read_root():
            if False:
                for i in range(10):
                    print('nop')
            pass

def test_invalid_response_model_in_responses_raises():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(FastAPIError):
        app = FastAPI()

        @app.get('/', responses={'500': {'model': NonPydanticModel}})
        def read_root():
            if False:
                i = 10
                return i + 15
            pass

def test_invalid_response_model_sub_type_in_responses_raises():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(FastAPIError):
        app = FastAPI()

        @app.get('/', responses={'500': {'model': List[NonPydanticModel]}})
        def read_root():
            if False:
                while True:
                    i = 10
            pass