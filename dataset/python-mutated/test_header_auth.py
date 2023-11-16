"""
.. module: security_monkey.tests.sso.header_auth
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Jordan Milne <jordan.milne@reddit.com>

"""
from mock import patch
from security_monkey.tests import SecurityMonkeyTestCase
from security_monkey.datastore import User

class HeaderAuthTestCase(SecurityMonkeyTestCase):

    def _get_user(self, email):
        if False:
            i = 10
            return i + 15
        return User.query.filter(User.email == email).scalar()

    def test_header_auth_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        with patch.dict(self.app.config, {'USE_HEADER_AUTH': False}):
            r = self.test_app.get('/login', headers={'Remote-User': 'foo@example.com'})
            self.assertFalse(self._get_user('foo@example.com'))
            self.assertEqual(r.status_code, 200)

    def test_header_auth_enabled(self):
        if False:
            print('Hello World!')
        with patch.dict(self.app.config, {'USE_HEADER_AUTH': True}):
            r = self.test_app.get('/login', headers={'Remote-User': 'foo@example.com'})
            user = self._get_user('foo@example.com')
            self.assertIsNotNone(user)
            self.assertEqual(user.role, 'View')
            self.assertEqual(r.status_code, 302)

    def test_header_auth_groups_used(self):
        if False:
            for i in range(10):
                print('nop')
        with patch.dict(self.app.config, {'USE_HEADER_AUTH': True, 'HEADER_AUTH_GROUPS_HEADER': 'Remote-Groups', 'ADMIN_GROUP': 'admingroup'}):
            r = self.test_app.get('/login', headers={'Remote-User': 'foo@example.com', 'Remote-Groups': 'foo,admingroup'})
            user = self._get_user('foo@example.com')
            self.assertIsNotNone(user)
            self.assertEqual(user.role, 'Admin')
            self.assertEqual(r.status_code, 302)