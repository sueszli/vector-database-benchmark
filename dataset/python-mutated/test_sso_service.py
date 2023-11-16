"""
.. module: security_monkey.tests.core.test_sso_service
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.tests import SecurityMonkeyTestCase
from security_monkey.sso.service import setup_user
from security_monkey.datastore import User
from security_monkey import db

class SSOServiceTestCase(SecurityMonkeyTestCase):

    def test_create_user(self):
        if False:
            print('Hello World!')
        existing_user = User(email='test@test.com', active=True, role='View')
        db.session.add(existing_user)
        db.session.commit()
        db.session.refresh(existing_user)
        user1 = setup_user('test@test.com')
        self.assertEqual(existing_user.id, user1.id)
        self.assertEqual(existing_user.role, user1.role)
        user2 = setup_user('test2@test.com')
        self.assertEqual(user2.email, 'test2@test.com')
        self.assertEqual(user2.role, 'View')
        self.app.config.update(ADMIN_GROUP='test_admin_group', JUSTIFY_GROUP='test_justify_group', VIEW_GROUP='test_view_group')
        admin_user = setup_user('admin@test.com', ['test_admin_group'])
        justify_user = setup_user('justifier@test.com', ['test_justify_group'])
        view_user = setup_user('viewer@test.com', ['test_view_group'])
        self.assertEqual(admin_user.role, 'Admin')
        self.assertEqual(justify_user.role, 'Justify')
        self.assertEqual(view_user.role, 'View')