import pytest
from superset.extensions import db, security_manager
from tests.integration_tests.test_app import app

@pytest.fixture()
def public_role_like_gamma():
    if False:
        return 10
    with app.app_context():
        app.config['PUBLIC_ROLE_LIKE'] = 'Gamma'
        security_manager.sync_role_definitions()
        yield
        security_manager.get_public_role().permissions = []
        db.session.commit()

@pytest.fixture()
def public_role_like_test_role():
    if False:
        while True:
            i = 10
    with app.app_context():
        app.config['PUBLIC_ROLE_LIKE'] = 'TestRole'
        security_manager.sync_role_definitions()
        yield
        security_manager.get_public_role().permissions = []
        db.session.commit()