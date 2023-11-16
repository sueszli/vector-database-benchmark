import uuid
import datetime as dt
from pylons import tmpl_context as c
from pylons import app_globals as g
from pylons import request
from r2.lib.cookies import Cookies, Cookie, upgrade_cookie_security, NEVER
from r2.models import Account, bcrypt_password, COOKIE_TIMESTAMP_FORMAT
from r2.tests import RedditTestCase

class TestCookieUpgrade(RedditTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        name = 'unit_tester_%s' % uuid.uuid4().hex
        self._password = uuid.uuid4().hex
        self._account = Account(name=name, password=bcrypt_password(self._password))
        self._account._id = 1337
        c.cookies = Cookies()
        c.secure = True
        c.user_is_loggedin = True
        c.user = self._account
        c.oauth_user = None
        request.method = 'POST'

    def tearDown(self):
        if False:
            print('Hello World!')
        c.cookies.clear()
        c.user_is_loggedin = False
        c.user = None

    def _setSessionCookie(self, days_old=0):
        if False:
            print('Hello World!')
        date = dt.datetime.now() - dt.timedelta(days=days_old)
        date_str = date.strftime(COOKIE_TIMESTAMP_FORMAT)
        session_cookie = self._account.make_cookie(date_str)
        c.cookies[g.login_cookie] = Cookie(value=session_cookie, dirty=False)

    def test_no_upgrade_loggedout(self):
        if False:
            return 10
        c.user_is_loggedin = False
        c.user = None
        self._setSessionCookie(days_old=60)
        upgrade_cookie_security()
        self.assertFalse(c.cookies[g.login_cookie].dirty)

    def test_no_upgrade_http(self):
        if False:
            while True:
                i = 10
        c.secure = False
        self._setSessionCookie(days_old=60)
        upgrade_cookie_security()
        self.assertFalse(c.cookies[g.login_cookie].dirty)

    def test_no_upgrade_no_cookie(self):
        if False:
            for i in range(10):
                print('nop')
        upgrade_cookie_security()
        self.assertFalse(g.login_cookie in c.cookies)

    def test_no_upgrade_oauth(self):
        if False:
            return 10
        c.oauth_user = self._account
        self._setSessionCookie(days_old=60)
        upgrade_cookie_security()
        self.assertFalse(c.cookies[g.login_cookie].dirty)

    def test_no_upgrade_gets(self):
        if False:
            for i in range(10):
                print('nop')
        request.method = 'GET'
        self._setSessionCookie(days_old=60)
        upgrade_cookie_security()
        self.assertFalse(c.cookies[g.login_cookie].dirty)

    def test_no_upgrade_secure_session(self):
        if False:
            while True:
                i = 10
        self._setSessionCookie(days_old=60)
        c.cookies['secure_session'] = Cookie(value='1')
        upgrade_cookie_security()
        self.assertFalse(c.cookies[g.login_cookie].dirty)

    def test_upgrade_posts(self):
        if False:
            while True:
                i = 10
        self._setSessionCookie(days_old=60)
        upgrade_cookie_security()
        self.assertTrue(c.cookies[g.login_cookie].dirty)
        self.assertTrue(c.cookies[g.login_cookie].secure)

    def test_cookie_unchanged(self):
        if False:
            while True:
                i = 10
        self._setSessionCookie(days_old=60)
        old_session = c.cookies[g.login_cookie].value
        upgrade_cookie_security()
        self.assertTrue(c.cookies[g.login_cookie].dirty)
        self.assertEqual(old_session, c.cookies[g.login_cookie].value)

    def test_remember_old_session(self):
        if False:
            print('Hello World!')
        self._setSessionCookie(days_old=60)
        upgrade_cookie_security()
        self.assertTrue(c.cookies[g.login_cookie].dirty)
        self.assertEqual(c.cookies[g.login_cookie].expires, NEVER)

    def test_dont_remember_recent_session(self):
        if False:
            return 10
        self._setSessionCookie(days_old=5)
        upgrade_cookie_security()
        self.assertTrue(c.cookies[g.login_cookie].dirty)
        self.assertNotEqual(c.cookies[g.login_cookie].expires, NEVER)