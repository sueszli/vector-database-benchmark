from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from .utils import needs_pydanticv1, needs_pydanticv2

@needs_pydanticv2
def test_pydanticv2():
    if False:
        while True:
            i = 10
    from pydantic import field_serializer

    class ModelWithDatetimeField(BaseModel):
        dt_field: datetime

        @field_serializer('dt_field')
        def serialize_datetime(self, dt_field: datetime):
            if False:
                while True:
                    i = 10
            return dt_field.replace(microsecond=0, tzinfo=timezone.utc).isoformat()
    app = FastAPI()
    model = ModelWithDatetimeField(dt_field=datetime(2019, 1, 1, 8))

    @app.get('/model', response_model=ModelWithDatetimeField)
    def get_model():
        if False:
            for i in range(10):
                print('nop')
        return model
    client = TestClient(app)
    with client:
        response = client.get('/model')
    assert response.json() == {'dt_field': '2019-01-01T08:00:00+00:00'}

@needs_pydanticv1
def test_pydanticv1():
    if False:
        for i in range(10):
            print('nop')

    class ModelWithDatetimeField(BaseModel):
        dt_field: datetime

        class Config:
            json_encoders = {datetime: lambda dt: dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat()}
    app = FastAPI()
    model = ModelWithDatetimeField(dt_field=datetime(2019, 1, 1, 8))

    @app.get('/model', response_model=ModelWithDatetimeField)
    def get_model():
        if False:
            for i in range(10):
                print('nop')
        return model
    client = TestClient(app)
    with client:
        response = client.get('/model')
    assert response.json() == {'dt_field': '2019-01-01T08:00:00+00:00'}