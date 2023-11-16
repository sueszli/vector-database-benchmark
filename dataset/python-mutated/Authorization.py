from datetime import datetime, timezone
from . import Framework

class Authorization(Framework.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.authorization = self.g.get_user().get_authorization(372259)

    def testAttributes(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.authorization.app.url, 'http://developer.github.com/v3/oauth/#oauth-authorizations-api')
        self.assertEqual(self.authorization.app.name, 'GitHub API')
        self.assertEqual(self.authorization.created_at, datetime(2012, 5, 22, 18, 3, 17, tzinfo=timezone.utc))
        self.assertEqual(self.authorization.id, 372259)
        self.assertEqual(self.authorization.note, None)
        self.assertEqual(self.authorization.note_url, None)
        self.assertEqual(self.authorization.scopes, [])
        self.assertEqual(self.authorization.token, '82459c4500086f8f0cc67d2936c17d1e27ad1c33')
        self.assertEqual(self.authorization.updated_at, datetime(2012, 5, 22, 18, 3, 17, tzinfo=timezone.utc))
        self.assertEqual(self.authorization.url, 'https://api.github.com/authorizations/372259')
        self.assertEqual(repr(self.authorization), 'Authorization(scopes=[])')
        self.assertEqual(repr(self.authorization.app), 'AuthorizationApplication(name="GitHub API")')

    def testEdit(self):
        if False:
            return 10
        self.authorization.edit()
        self.assertEqual(self.authorization.scopes, [])
        self.authorization.edit(scopes=['user'])
        self.assertEqual(self.authorization.scopes, ['user'])
        self.authorization.edit(add_scopes=['repo'])
        self.assertEqual(self.authorization.scopes, ['user', 'repo'])
        self.authorization.edit(remove_scopes=['repo'])
        self.assertEqual(self.authorization.scopes, ['user'])
        self.assertEqual(self.authorization.note, None)
        self.assertEqual(self.authorization.note_url, None)
        self.authorization.edit(note='Note created by PyGithub', note_url='http://vincent-jacques.net/PyGithub')
        self.assertEqual(self.authorization.note, 'Note created by PyGithub')
        self.assertEqual(self.authorization.note_url, 'http://vincent-jacques.net/PyGithub')

    def testDelete(self):
        if False:
            while True:
                i = 10
        self.authorization.delete()