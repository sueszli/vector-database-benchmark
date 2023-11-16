from datetime import datetime, timezone
from . import Framework

class Reaction(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.reactions = self.g.get_user('PyGithub').get_repo('PyGithub').get_issue(28).get_reactions()

    def testAttributes(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.reactions[0].content, '+1')
        self.assertEqual(self.reactions[0].created_at, datetime(2017, 12, 5, 1, 59, 33, tzinfo=timezone.utc))
        self.assertEqual(self.reactions[0].id, 16916340)
        self.assertEqual(self.reactions[0].user.login, 'nicolastrres')
        self.assertEqual(self.reactions[0].__repr__(), 'Reaction(user=NamedUser(login="nicolastrres"), id=16916340)')

    def testDelete(self):
        if False:
            i = 10
            return i + 15
        self.reactions[0].delete()