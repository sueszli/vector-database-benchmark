import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
from .main import app, get_session

@pytest.fixture(name='session')
def session_fixture():
    if False:
        i = 10
        return i + 15
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

@pytest.fixture(name='client')
def client_fixture(session: Session):
    if False:
        while True:
            i = 10

    def get_session_override():
        if False:
            while True:
                i = 10
        return session
    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

def test_create_hero(client: TestClient):
    if False:
        while True:
            i = 10
    response = client.post('/heroes/', json={'name': 'Deadpond', 'secret_name': 'Dive Wilson'})
    data = response.json()
    assert response.status_code == 200
    assert data['name'] == 'Deadpond'
    assert data['secret_name'] == 'Dive Wilson'
    assert data['age'] is None
    assert data['id'] is not None