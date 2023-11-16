from unittest.mock import patch
from flask_appbuilder import ModelView
from flask_appbuilder.exceptions import PasswordComplexityValidationError
from flask_appbuilder.models.sqla.filters import FilterEqual
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.sqla.models import User
from tests.base import BaseMVCTestCase
from tests.const import INVALID_LOGIN_STRING, PASSWORD_ADMIN, PASSWORD_READONLY, USERNAME_ADMIN, USERNAME_READONLY
from tests.fixtures.data_models import model1_data
from tests.sqla.models import Model1, Model2
PASSWORD_COMPLEXITY_ERROR = 'Must have at least two capital letters, one special character, two digits, three lower case letters and a minimal length of 10'

def custom_password_validator(password: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    A simplistic example for a password validator\n    '
    if password != 'password':
        raise PasswordComplexityValidationError('Password must be password')

class MVCSecurityTestCase(BaseMVCTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.client = self.app.test_client()

        class Model2View(ModelView):
            datamodel = SQLAInterface(Model2)
            list_columns = ['field_integer', 'field_float', 'field_string', 'field_method', 'group.field_string']
            edit_form_query_rel_fields = {'group': [['field_string', FilterEqual, 'test1']]}
            add_form_query_rel_fields = {'group': [['field_string', FilterEqual, 'test0']]}
            order_columns = ['field_string', 'group.field_string']
        self.appbuilder.add_view(Model2View, 'Model2')

        class Model1View(ModelView):
            datamodel = SQLAInterface(Model1)
            related_views = [Model2View]
            list_columns = ['field_string', 'field_integer']
        self.appbuilder.add_view(Model1View, 'Model1', category='Model1')

    def test_sec_login(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Security Login, Logout, invalid login, invalid access\n        '
        rv = self.client.get('/model1view/list/')
        self.assertEqual(rv.status_code, 302)
        rv = self.client.get('/model2view/list/')
        self.assertEqual(rv.status_code, 302)
        self.browser_login(self.client, USERNAME_ADMIN, PASSWORD_ADMIN)
        rv = self.client.get('/model1view/list/')
        self.assertEqual(rv.status_code, 200)
        rv = self.client.get('/model2view/list/')
        self.assertEqual(rv.status_code, 200)
        self.browser_logout(self.client)
        rv = self.client.get('/model1view/list/')
        self.assertEqual(rv.status_code, 302)
        rv = self.client.get('/model2view/list/')
        self.assertEqual(rv.status_code, 302)
        rv = self.browser_login(self.client, USERNAME_ADMIN, 'wrong_password')
        data = rv.data.decode('utf-8')
        self.assertIn(INVALID_LOGIN_STRING, data)

    def test_db_login_no_next_url(self):
        if False:
            print('Hello World!')
        '\n        Test Security no next URL\n        '
        self.browser_logout(self.client)
        response = self.browser_login(self.client, USERNAME_ADMIN, PASSWORD_ADMIN, follow_redirects=False)
        assert response.location == '/'

    def test_db_login_valid_next_url(self):
        if False:
            print('Hello World!')
        '\n        Test Security valid partial next URL\n        '
        self.browser_logout(self.client)
        response = self.browser_login(self.client, USERNAME_ADMIN, PASSWORD_ADMIN, next_url='/users/list/', follow_redirects=False)
        assert response.location == '/users/list/'

    def test_db_login_valid_http_scheme_url(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test Security valid http scheme next URL\n        '
        self.browser_logout(self.client)
        response = self.browser_login(self.client, USERNAME_ADMIN, PASSWORD_ADMIN, next_url='http://localhost/path', follow_redirects=False)
        assert response.location == 'http://localhost/path'

    def test_db_login_valid_https_scheme_url(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Security valid https scheme next URL\n        '
        self.browser_logout(self.client)
        response = self.browser_login(self.client, USERNAME_ADMIN, PASSWORD_ADMIN, next_url='https://localhost/path', follow_redirects=False)
        assert response.location == 'https://localhost/path'

    def test_db_login_invalid_external_next_url(self):
        if False:
            while True:
                i = 10
        '\n        Test Security invalid external next URL\n        '
        self.browser_logout(self.client)
        response = self.browser_login(self.client, USERNAME_ADMIN, PASSWORD_ADMIN, next_url='https://google.com', follow_redirects=False)
        assert response.location == '/'

    def test_db_login_invalid_scheme_next_url(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Security invalid scheme next URL\n        '
        self.browser_logout(self.client)
        response = self.browser_login(self.client, USERNAME_ADMIN, PASSWORD_ADMIN, next_url='ftp://sample', follow_redirects=False)
        assert response.location == '/'

    def test_db_login_invalid_localhost_file_next_url(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test Security invalid path to localhost file next URL\n        '
        self.browser_logout(self.client)
        response = self.browser_login(self.client, USERNAME_ADMIN, PASSWORD_ADMIN, next_url='file:///path', follow_redirects=False)
        assert response.location == '/'

    def test_db_login_invalid_no_netloc_with_scheme_next_url(self):
        if False:
            while True:
                i = 10
        '\n        Test Security invalid next URL with no netloc but with scheme\n        '
        self.browser_logout(self.client)
        response = self.browser_login(self.client, USERNAME_ADMIN, PASSWORD_ADMIN, next_url='http:///sample.com ', follow_redirects=False)
        assert response.location == '/'

    def test_db_login_invalid_control_characters_next_url(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test Security invalid next URL with control characters\n        '
        self.browser_logout(self.client)
        response = self.browser_login(self.client, USERNAME_ADMIN, PASSWORD_ADMIN, next_url='\x01' + 'sample.com', follow_redirects=False)
        assert response.location == '/'

    def test_db_login_failed_keep_next_url(self):
        if False:
            print('Hello World!')
        '\n        Test Security Keeping next url after failed login attempt\n        '
        self.browser_logout(self.client)
        response = self.browser_login(self.client, USERNAME_ADMIN, f'wrong_{PASSWORD_ADMIN}', next_url='/users/list/', follow_redirects=False)
        response = self.client.post(response.location, data=dict(username=USERNAME_ADMIN, password=PASSWORD_ADMIN), follow_redirects=False)
        assert response.location == '/users/list/'

    def test_auth_builtin_roles(self):
        if False:
            print('Hello World!')
        '\n        Test Security builtin roles readonly\n        '
        client = self.app.test_client()
        self.browser_login(client, USERNAME_READONLY, PASSWORD_READONLY)
        with model1_data(self.appbuilder.session, 1) as model_data:
            model_id = model_data[0].id
            rv = client.get('/model1view/list/')
            self.assertEqual(rv.status_code, 200)
            rv = client.get(f'/model1view/show/{model_id}')
            self.assertEqual(rv.status_code, 200)
            rv = client.get(f'/model1view/edit/{model_id}')
            self.assertEqual(rv.status_code, 302)
            rv = client.get(f'/model1view/delete/{model_id}')
            self.assertEqual(rv.status_code, 302)

    def test_sec_reset_password(self):
        if False:
            return 10
        '\n        Test Security reset password\n        '
        client = self.app.test_client()
        admin_user = self.appbuilder.sm.find_user(username=USERNAME_ADMIN)
        rv = client.get(f'/users/action/resetmypassword/{admin_user.id}', follow_redirects=True)
        self.assertEqual(rv.status_code, 404)
        _ = self.browser_login(client, USERNAME_ADMIN, PASSWORD_ADMIN)
        rv = client.get(f'/users/action/resetmypassword/{admin_user.id}', follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Reset Password Form', data)
        rv = client.post('/resetmypassword/form', data=dict(password='password', conf_password='password'), follow_redirects=True)
        self.assertEqual(rv.status_code, 200)
        self.browser_logout(client)
        self.browser_login(client, USERNAME_ADMIN, 'password')
        rv = client.post('/resetmypassword/form', data=dict(password=PASSWORD_ADMIN, conf_password=PASSWORD_ADMIN), follow_redirects=True)
        self.assertEqual(rv.status_code, 200)
        rv = client.get(f'/users/action/resetpasswords/{admin_user.id}', follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Reset Password Form', data)
        rv = client.post('/resetmypassword/form', data=dict(password=PASSWORD_ADMIN, conf_password=PASSWORD_ADMIN), follow_redirects=True)
        self.assertEqual(rv.status_code, 200)

    def test_sec_reset_password_default_complexity(self):
        if False:
            while True:
                i = 10
        '\n        Test Security reset password with default complexity\n        '
        client = self.app.test_client()
        self.app.config['FAB_PASSWORD_COMPLEXITY_ENABLED'] = True
        _ = self.browser_login(client, USERNAME_ADMIN, PASSWORD_ADMIN)
        rv = client.get('/users/action/resetmypassword/1', follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Reset Password Form', data)
        rv = client.post('/resetmypassword/form', data=dict(password='password', conf_password='password'), follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn(PASSWORD_COMPLEXITY_ERROR, data)
        rv = client.post('/resetmypassword/form', data=dict(password='PAssword123!', conf_password='PAssword123!'), follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertNotIn(PASSWORD_COMPLEXITY_ERROR, data)
        self.app.config['FAB_PASSWORD_COMPLEXITY_ENABLED'] = False
        _ = client.post('/resetmypassword/form', data=dict(password='password', conf_password='password'), follow_redirects=True)
        self.browser_logout(client)

    def test_sec_reset_password_custom_complexity(self):
        if False:
            while True:
                i = 10
        '\n        Test Security reset password with custom complexity\n        '
        client = self.app.test_client()
        self.app.config['FAB_PASSWORD_COMPLEXITY_ENABLED'] = True
        self.app.config['FAB_PASSWORD_COMPLEXITY_VALIDATOR'] = custom_password_validator
        _ = self.browser_login(client, USERNAME_ADMIN, PASSWORD_ADMIN)
        rv = client.get('/users/action/resetmypassword/1', follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Reset Password Form', data)
        rv = client.post('/resetmypassword/form', data=dict(password='123', conf_password='123'), follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Password must be password', data)
        rv = client.post('/resetmypassword/form', data=dict(password='password', conf_password='password'), follow_redirects=True)
        self.browser_logout(client)

    def test_register_user(self):
        if False:
            while True:
                i = 10
        '\n        Test register user\n        '
        client = self.app.test_client()
        _ = self.browser_login(client, USERNAME_ADMIN, PASSWORD_ADMIN)
        rv = client.get('/users/add', follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Add User', data)
        rv = client.post('/users/add', data=dict(first_name='first', last_name='last', username='from test 1-1', email='test1@fromtest1.com', roles=[1], password='password', conf_password='password'), follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Added Row', data)
        rv = client.get('/users/add', follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Add User', data)
        rv = client.post('/users/add', data=dict(first_name='first', last_name='last', username='from test 2-1', email='test2@fromtest2.com', roles=[], password='password', conf_password='password'), follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertNotIn('Added Row', data)
        self.assertIn('This field is required', data)
        self.browser_logout(client)
        user = self.db.session.query(User).filter(User.username == 'from test 1-1').one_or_none()
        self.db.session.delete(user)
        self.db.session.commit()

    def test_edit_user(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test edit user\n        '
        client = self.app.test_client()
        _ = self.browser_login(client, USERNAME_ADMIN, PASSWORD_ADMIN)
        _tmp_user = self.create_user(self.appbuilder, 'tmp_user', 'password1', '', first_name='tmp', last_name='user', email='tmp@fab.org', role_names=['Admin'])
        rv = client.get(f'/users/edit/{_tmp_user.id}', follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Edit User', data)
        rv = client.post(f'/users/edit/{_tmp_user.id}', data=dict(first_name=_tmp_user.first_name, last_name=_tmp_user.last_name, username=_tmp_user.username, email='changed@changed.org', roles=_tmp_user.roles[0].id), follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Changed Row', data)
        user = self.db.session.query(User).filter(User.username == _tmp_user.username).one_or_none()
        assert user.email == 'changed@changed.org'
        self.db.session.delete(user)
        self.db.session.commit()

    def test_edit_user_email_validation(self):
        if False:
            print('Hello World!')
        '\n        Test edit user with email not null validation\n        '
        client = self.app.test_client()
        _ = self.browser_login(client, USERNAME_ADMIN, PASSWORD_ADMIN)
        read_ony_user: User = self.db.session.query(User).filter(User.username == USERNAME_READONLY).one_or_none()
        rv = client.get(f'/users/edit/{read_ony_user.id}', follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Edit User', data)
        rv = client.post(f'/users/edit/{read_ony_user.id}', data=dict(first_name=read_ony_user.first_name, last_name=read_ony_user.last_name, username=read_ony_user.username, email=None, roles=read_ony_user.roles[0].id), follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('This field is required', data)

    def test_edit_user_db_fail(self):
        if False:
            while True:
                i = 10
        '\n        Test edit user with DB fail\n        '
        client = self.app.test_client()
        _ = self.browser_login(client, USERNAME_ADMIN, PASSWORD_ADMIN)
        read_ony_user: User = self.db.session.query(User).filter(User.username == USERNAME_READONLY).one_or_none()
        rv = client.get(f'/users/edit/{read_ony_user.id}', follow_redirects=True)
        data = rv.data.decode('utf-8')
        self.assertIn('Edit User', data)
        with patch.object(self.appbuilder.session, 'merge') as mock_merge:
            with patch.object(self.appbuilder.sm, 'has_access', return_value=True) as _:
                mock_merge.side_effect = Exception('BANG!')
                rv = client.post(f'/users/edit/{read_ony_user.id}', data=dict(first_name=read_ony_user.first_name, last_name=read_ony_user.last_name, username=read_ony_user.username, email='changed@changed.org', roles=read_ony_user.roles[0].id), follow_redirects=True)
                data = rv.data.decode('utf-8')
                self.assertIn('Database Error', data)