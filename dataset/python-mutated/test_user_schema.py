from __future__ import annotations
import pytest
from airflow.api_connexion.schemas.user_schema import user_collection_item_schema, user_schema
from airflow.auth.managers.fab.models import User
from airflow.utils import timezone
from tests.test_utils.api_connexion_utils import create_role, delete_role
TEST_EMAIL = 'test@example.org'
DEFAULT_TIME = '2021-01-09T13:59:56.336000+00:00'
pytestmark = pytest.mark.db_test

@pytest.fixture(scope='module')
def configured_app(minimal_app_for_api):
    if False:
        i = 10
        return i + 15
    app = minimal_app_for_api
    create_role(app, name='TestRole', permissions=[])
    yield app
    delete_role(app, 'TestRole')

class TestUserBase:

    @pytest.fixture(autouse=True)
    def setup_attrs(self, configured_app) -> None:
        if False:
            while True:
                i = 10
        self.app = configured_app
        self.client = self.app.test_client()
        self.role = self.app.appbuilder.sm.find_role('TestRole')
        self.session = self.app.appbuilder.get_session

    def teardown_method(self):
        if False:
            i = 10
            return i + 15
        user = self.session.query(User).filter(User.email == TEST_EMAIL).first()
        if user:
            self.session.delete(user)
            self.session.commit()

class TestUserCollectionItemSchema(TestUserBase):

    def test_serialize(self):
        if False:
            while True:
                i = 10
        user_model = User(first_name='Foo', last_name='Bar', username='test', password='test', email=TEST_EMAIL, roles=[self.role], created_on=timezone.parse(DEFAULT_TIME), changed_on=timezone.parse(DEFAULT_TIME))
        self.session.add(user_model)
        self.session.commit()
        user = self.session.query(User).filter(User.email == TEST_EMAIL).first()
        deserialized_user = user_collection_item_schema.dump(user)
        assert deserialized_user == {'created_on': DEFAULT_TIME, 'email': 'test@example.org', 'changed_on': DEFAULT_TIME, 'active': None, 'last_login': None, 'last_name': 'Bar', 'fail_login_count': None, 'first_name': 'Foo', 'username': 'test', 'login_count': None, 'roles': [{'name': 'TestRole'}]}

class TestUserSchema(TestUserBase):

    def test_serialize(self):
        if False:
            print('Hello World!')
        user_model = User(first_name='Foo', last_name='Bar', username='test', password='test', email=TEST_EMAIL, created_on=timezone.parse(DEFAULT_TIME), changed_on=timezone.parse(DEFAULT_TIME))
        self.session.add(user_model)
        self.session.commit()
        user = self.session.query(User).filter(User.email == TEST_EMAIL).first()
        deserialized_user = user_schema.dump(user)
        assert deserialized_user == {'roles': [], 'created_on': DEFAULT_TIME, 'email': 'test@example.org', 'changed_on': DEFAULT_TIME, 'active': None, 'last_login': None, 'last_name': 'Bar', 'fail_login_count': None, 'first_name': 'Foo', 'username': 'test', 'login_count': None}

    def test_deserialize_user(self):
        if False:
            i = 10
            return i + 15
        user_dump = {'roles': [{'name': 'TestRole'}], 'email': 'test@example.org', 'last_name': 'Bar', 'first_name': 'Foo', 'username': 'test', 'password': 'test'}
        result = user_schema.load(user_dump)
        assert result == {'roles': [{'name': 'TestRole'}], 'email': 'test@example.org', 'last_name': 'Bar', 'first_name': 'Foo', 'username': 'test', 'password': 'test'}