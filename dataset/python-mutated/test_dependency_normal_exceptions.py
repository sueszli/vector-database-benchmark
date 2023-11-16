import pytest
from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
initial_fake_database = {'rick': 'Rick Sanchez'}
fake_database = initial_fake_database.copy()
initial_state = {'except': False, 'finally': False}
state = initial_state.copy()
app = FastAPI()

async def get_database():
    temp_database = fake_database.copy()
    try:
        yield temp_database
        fake_database.update(temp_database)
    except HTTPException:
        state['except'] = True
    finally:
        state['finally'] = True

@app.put('/invalid-user/{user_id}')
def put_invalid_user(user_id: str, name: str=Body(), db: dict=Depends(get_database)):
    if False:
        for i in range(10):
            print('nop')
    db[user_id] = name
    raise HTTPException(status_code=400, detail='Invalid user')

@app.put('/user/{user_id}')
def put_user(user_id: str, name: str=Body(), db: dict=Depends(get_database)):
    if False:
        i = 10
        return i + 15
    db[user_id] = name
    return {'message': 'OK'}

@pytest.fixture(autouse=True)
def reset_state_and_db():
    if False:
        return 10
    global fake_database
    global state
    fake_database = initial_fake_database.copy()
    state = initial_state.copy()
client = TestClient(app)

def test_dependency_gets_exception():
    if False:
        for i in range(10):
            print('nop')
    assert state['except'] is False
    assert state['finally'] is False
    response = client.put('/invalid-user/rick', json='Morty')
    assert response.status_code == 400, response.text
    assert response.json() == {'detail': 'Invalid user'}
    assert state['except'] is True
    assert state['finally'] is True
    assert fake_database['rick'] == 'Rick Sanchez'

def test_dependency_no_exception():
    if False:
        return 10
    assert state['except'] is False
    assert state['finally'] is False
    response = client.put('/user/rick', json='Morty')
    assert response.status_code == 200, response.text
    assert response.json() == {'message': 'OK'}
    assert state['except'] is False
    assert state['finally'] is True
    assert fake_database['rick'] == 'Morty'