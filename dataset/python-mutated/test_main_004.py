from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
from .main import app, get_session

def test_create_hero():
    if False:
        while True:
            i = 10
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:

        def get_session_override():
            if False:
                print('Hello World!')
            return session
        app.dependency_overrides[get_session] = get_session_override
        client = TestClient(app)
        response = client.post('/heroes/', json={'name': 'Deadpond', 'secret_name': 'Dive Wilson'})
        app.dependency_overrides.clear()
        data = response.json()
        assert response.status_code == 200
        assert data['name'] == 'Deadpond'
        assert data['secret_name'] == 'Dive Wilson'
        assert data['age'] is None
        assert data['id'] is not None