import uuid
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from .utils import needs_pydanticv1, needs_pydanticv2

class MyUuid:

    def __init__(self, uuid_string: str):
        if False:
            return 10
        self.uuid = uuid_string

    def __str__(self):
        if False:
            return 10
        return self.uuid

    @property
    def __class__(self):
        if False:
            return 10
        return uuid.UUID

    @property
    def __dict__(self):
        if False:
            return 10
        'Spoof a missing __dict__ by raising TypeError, this is how\n        asyncpg.pgroto.pgproto.UUID behaves'
        raise TypeError('vars() argument must have __dict__ attribute')

@needs_pydanticv2
def test_pydanticv2():
    if False:
        while True:
            i = 10
    from pydantic import field_serializer
    app = FastAPI()

    @app.get('/fast_uuid')
    def return_fast_uuid():
        if False:
            i = 10
            return i + 15
        asyncpg_uuid = MyUuid('a10ff360-3b1e-4984-a26f-d3ab460bdb51')
        assert isinstance(asyncpg_uuid, uuid.UUID)
        assert type(asyncpg_uuid) != uuid.UUID
        with pytest.raises(TypeError):
            vars(asyncpg_uuid)
        return {'fast_uuid': asyncpg_uuid}

    class SomeCustomClass(BaseModel):
        model_config = {'arbitrary_types_allowed': True}
        a_uuid: MyUuid

        @field_serializer('a_uuid')
        def serialize_a_uuid(self, v):
            if False:
                while True:
                    i = 10
            return str(v)

    @app.get('/get_custom_class')
    def return_some_user():
        if False:
            while True:
                i = 10
        return SomeCustomClass(a_uuid=MyUuid('b8799909-f914-42de-91bc-95c819218d01'))
    client = TestClient(app)
    with client:
        response_simple = client.get('/fast_uuid')
        response_pydantic = client.get('/get_custom_class')
    assert response_simple.json() == {'fast_uuid': 'a10ff360-3b1e-4984-a26f-d3ab460bdb51'}
    assert response_pydantic.json() == {'a_uuid': 'b8799909-f914-42de-91bc-95c819218d01'}

@needs_pydanticv1
def test_pydanticv1():
    if False:
        for i in range(10):
            print('nop')
    app = FastAPI()

    @app.get('/fast_uuid')
    def return_fast_uuid():
        if False:
            print('Hello World!')
        asyncpg_uuid = MyUuid('a10ff360-3b1e-4984-a26f-d3ab460bdb51')
        assert isinstance(asyncpg_uuid, uuid.UUID)
        assert type(asyncpg_uuid) != uuid.UUID
        with pytest.raises(TypeError):
            vars(asyncpg_uuid)
        return {'fast_uuid': asyncpg_uuid}

    class SomeCustomClass(BaseModel):

        class Config:
            arbitrary_types_allowed = True
            json_encoders = {uuid.UUID: str}
        a_uuid: MyUuid

    @app.get('/get_custom_class')
    def return_some_user():
        if False:
            return 10
        return SomeCustomClass(a_uuid=MyUuid('b8799909-f914-42de-91bc-95c819218d01'))
    client = TestClient(app)
    with client:
        response_simple = client.get('/fast_uuid')
        response_pydantic = client.get('/get_custom_class')
    assert response_simple.json() == {'fast_uuid': 'a10ff360-3b1e-4984-a26f-d3ab460bdb51'}
    assert response_pydantic.json() == {'a_uuid': 'b8799909-f914-42de-91bc-95c819218d01'}