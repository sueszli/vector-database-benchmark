from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient
from sqlalchemy.sql.elements import quoted_name
app = FastAPI(default_response_class=ORJSONResponse)

@app.get('/orjson_non_str_keys')
def get_orjson_non_str_keys():
    if False:
        while True:
            i = 10
    key = quoted_name(value='msg', quote=False)
    return {key: 'Hello World', 1: 1}
client = TestClient(app)

def test_orjson_non_str_keys():
    if False:
        i = 10
        return i + 15
    with client:
        response = client.get('/orjson_non_str_keys')
    assert response.json() == {'msg': 'Hello World', '1': 1}