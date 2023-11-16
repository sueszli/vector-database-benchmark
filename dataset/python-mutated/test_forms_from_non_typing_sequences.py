from fastapi import FastAPI, Form
from fastapi.testclient import TestClient
app = FastAPI()

@app.post('/form/python-list')
def post_form_param_list(items: list=Form()):
    if False:
        i = 10
        return i + 15
    return items

@app.post('/form/python-set')
def post_form_param_set(items: set=Form()):
    if False:
        print('Hello World!')
    return items

@app.post('/form/python-tuple')
def post_form_param_tuple(items: tuple=Form()):
    if False:
        i = 10
        return i + 15
    return items
client = TestClient(app)

def test_python_list_param_as_form():
    if False:
        print('Hello World!')
    response = client.post('/form/python-list', data={'items': ['first', 'second', 'third']})
    assert response.status_code == 200, response.text
    assert response.json() == ['first', 'second', 'third']

def test_python_set_param_as_form():
    if False:
        return 10
    response = client.post('/form/python-set', data={'items': ['first', 'second', 'third']})
    assert response.status_code == 200, response.text
    assert set(response.json()) == {'first', 'second', 'third'}

def test_python_tuple_param_as_form():
    if False:
        for i in range(10):
            print('nop')
    response = client.post('/form/python-tuple', data={'items': ['first', 'second', 'third']})
    assert response.status_code == 200, response.text
    assert response.json() == ['first', 'second', 'third']