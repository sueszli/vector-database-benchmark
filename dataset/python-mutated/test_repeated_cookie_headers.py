from fastapi import Depends, FastAPI, Response
from fastapi.testclient import TestClient
app = FastAPI()

def set_cookie(*, response: Response):
    if False:
        for i in range(10):
            print('nop')
    response.set_cookie('cookie-name', 'cookie-value')
    return {}

def set_indirect_cookie(*, dep: str=Depends(set_cookie)):
    if False:
        i = 10
        return i + 15
    return dep

@app.get('/directCookie')
def get_direct_cookie(dep: str=Depends(set_cookie)):
    if False:
        print('Hello World!')
    return {'dep': dep}

@app.get('/indirectCookie')
def get_indirect_cookie(dep: str=Depends(set_indirect_cookie)):
    if False:
        return 10
    return {'dep': dep}
client = TestClient(app)

def test_cookie_is_set_once():
    if False:
        for i in range(10):
            print('nop')
    direct_response = client.get('/directCookie')
    indirect_response = client.get('/indirectCookie')
    assert direct_response.headers['set-cookie'] == indirect_response.headers['set-cookie']