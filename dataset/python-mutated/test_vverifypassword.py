import uuid
import unittest
from pylons import tmpl_context as c
from webob.exc import HTTPException
from r2.tests import RedditTestCase
from r2.lib.db.thing import NotFound
from r2.lib.errors import errors, ErrorSet, UserRequiredException
from r2.lib.validator import VVerifyPassword
from r2.models import Account, AccountExists, bcrypt_password

class TestVVerifyPassword(unittest.TestCase):
    """Test that only the current user's password satisfies VVerifyPassword"""

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        name = 'unit_tester_%s' % uuid.uuid4().hex
        cls._password = uuid.uuid4().hex
        cls._account = Account(name=name, password=bcrypt_password(cls._password))

    def setUp(self):
        if False:
            print('Hello World!')
        c.user_is_loggedin = True
        c.user = self._account

    def _checkFails(self, password, fatal=False, error=errors.WRONG_PASSWORD):
        if False:
            i = 10
            return i + 15
        c.errors = ErrorSet()
        validator = VVerifyPassword('dummy', fatal=fatal)
        if fatal:
            try:
                validator.run(password)
            except HTTPException:
                return True
            return False
        else:
            validator.run(password)
            return validator.has_errors or c.errors.get((error, None))

    def test_loggedout(self):
        if False:
            i = 10
            return i + 15
        c.user = ''
        c.user_is_loggedin = False
        self.assertRaises(UserRequiredException, self._checkFails, 'dummy')

    def test_right_password(self):
        if False:
            return 10
        self.assertFalse(self._checkFails(self._password, fatal=False))
        self.assertFalse(self._checkFails(self._password, fatal=True))

    def test_wrong_password(self):
        if False:
            print('Hello World!')
        bad_pass = '~' + self._password[1:]
        self.assertTrue(self._checkFails(bad_pass, fatal=False))
        self.assertTrue(self._checkFails(bad_pass, fatal=True))

    def test_no_password(self):
        if False:
            return 10
        self.assertTrue(self._checkFails(None, fatal=False))
        self.assertTrue(self._checkFails(None, fatal=True))
        self.assertTrue(self._checkFails('', fatal=False))
        self.assertTrue(self._checkFails('', fatal=True))