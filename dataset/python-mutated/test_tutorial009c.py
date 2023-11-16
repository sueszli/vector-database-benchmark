from fastapi.testclient import TestClient
from docs_src.custom_response.tutorial009c import app
client = TestClient(app)

def test_get():
    if False:
        i = 10
        return i + 15
    response = client.get('/')
    assert response.content == b'{\n  "message": "Hello World"\n}'