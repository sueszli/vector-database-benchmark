from r2.tests import RedditControllerTestCase
from r2.lib.errors import error_list
from r2.lib.unicode import _force_unicode
from r2.models import Subreddit
from common import LoginRegBase

class PostLoginRegTests(LoginRegBase, RedditControllerTestCase):
    CONTROLLER = 'post'
    ACTIONS = {'register': 'reg'}

    def setUp(self):
        if False:
            print('Hello World!')
        super(PostLoginRegTests, self).setUp()
        self.autopatch(Subreddit, '_byID', return_value=[])
        self.dest = '/foo'

    def assert_success(self, res):
        if False:
            print('Hello World!')
        self.assertEqual(res.status, 302)
        self.assert_headers(res, 'Location', lambda value: value.endswith(self.dest))
        self.assert_headers(res, 'Set-Cookie', lambda value: value.startswith('reddit_session='))

    def assert_failure(self, res, code=None):
        if False:
            return 10
        self.assertEqual(res.status, 200)
        if code != 'BAD_CAPTCHA':
            self.assertTrue(error_list[code] in _force_unicode(res.body))

    def make_qs(self, **kw):
        if False:
            while True:
                i = 10
        kw['dest'] = self.dest
        return super(PostLoginRegTests, self).make_qs(**kw)