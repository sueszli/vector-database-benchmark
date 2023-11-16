from datetime import datetime, timezone
from . import Framework

class Hook(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.hook = self.g.get_user().get_repo('PyGithub').get_hook(257993)

    def testAttributes(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.hook.active)
        self.assertEqual(self.hook.config, {'url': 'http://foobar.com'})
        self.assertEqual(self.hook.created_at, datetime(2012, 5, 19, 6, 1, 45, tzinfo=timezone.utc))
        self.assertEqual(self.hook.events, ['push'])
        self.assertEqual(self.hook.id, 257993)
        self.assertEqual(self.hook.last_response.status, 'ok')
        self.assertEqual(self.hook.last_response.message, 'OK')
        self.assertEqual(self.hook.last_response.code, 200)
        self.assertEqual(self.hook.name, 'web')
        self.assertEqual(self.hook.updated_at, datetime(2012, 5, 29, 18, 49, 47, tzinfo=timezone.utc))
        self.assertEqual(self.hook.url, 'https://api.github.com/repos/jacquev6/PyGithub/hooks/257993')
        self.assertEqual(self.hook.test_url, 'https://api.github.com/repos/jacquev6/PyGithub/hooks/257993/tests')
        self.assertEqual(self.hook.ping_url, 'https://api.github.com/repos/jacquev6/PyGithub/hooks/257993/pings')
        self.assertEqual(repr(self.hook), 'Hook(url="https://api.github.com/repos/jacquev6/PyGithub/hooks/257993", id=257993)')
        self.assertEqual(repr(self.hook.last_response), 'HookResponse(status="ok")')

    def testEditWithMinimalParameters(self):
        if False:
            while True:
                i = 10
        self.hook.edit('web', {'url': 'http://foobar.com/hook'})
        self.assertEqual(self.hook.config, {'url': 'http://foobar.com/hook'})
        self.assertEqual(self.hook.updated_at, datetime(2012, 5, 19, 5, 8, 16, tzinfo=timezone.utc))

    def testDelete(self):
        if False:
            print('Hello World!')
        self.hook.delete()

    def testTest(self):
        if False:
            return 10
        self.hook.test()

    def testPing(self):
        if False:
            print('Hello World!')
        self.hook.ping()

    def testEditWithAllParameters(self):
        if False:
            return 10
        self.hook.edit('web', {'url': 'http://foobar.com'}, events=['fork', 'push'])
        self.assertEqual(self.hook.events, ['fork', 'push'])
        self.hook.edit('web', {'url': 'http://foobar.com'}, add_events=['push'])
        self.assertEqual(self.hook.events, ['fork', 'push'])
        self.hook.edit('web', {'url': 'http://foobar.com'}, remove_events=['fork'])
        self.assertEqual(self.hook.events, ['push'])
        self.hook.edit('web', {'url': 'http://foobar.com'}, active=True)
        self.assertTrue(self.hook.active)