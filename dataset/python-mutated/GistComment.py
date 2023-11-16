from datetime import datetime, timezone
from . import Framework

class GistComment(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.comment = self.g.get_gist('2729810').get_comment(323629)

    def testAttributes(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.comment.body, 'Comment created by PyGithub')
        self.assertEqual(self.comment.created_at, datetime(2012, 5, 19, 7, 7, 57, tzinfo=timezone.utc))
        self.assertEqual(self.comment.id, 323629)
        self.assertEqual(self.comment.updated_at, datetime(2012, 5, 19, 7, 7, 57, tzinfo=timezone.utc))
        self.assertEqual(self.comment.url, 'https://api.github.com/gists/2729810/comments/323629')
        self.assertEqual(self.comment.user.login, 'jacquev6')
        self.assertEqual(repr(self.comment), 'GistComment(user=NamedUser(login="jacquev6"), id=323629)')

    def testEdit(self):
        if False:
            return 10
        self.comment.edit('Comment edited by PyGithub')
        self.assertEqual(self.comment.body, 'Comment edited by PyGithub')
        self.assertEqual(self.comment.updated_at, datetime(2012, 5, 19, 7, 12, 32, tzinfo=timezone.utc))

    def testDelete(self):
        if False:
            for i in range(10):
                print('nop')
        self.comment.delete()