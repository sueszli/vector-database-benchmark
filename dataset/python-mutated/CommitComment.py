from datetime import datetime, timezone
from . import Framework

class CommitComment(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.comment = self.g.get_user().get_repo('PyGithub').get_comment(1361949)

    def testAttributes(self):
        if False:
            return 10
        self.assertEqual(self.comment.body, 'Comment created by PyGithub')
        self.assertEqual(self.comment.commit_id, '6945921c529be14c3a8f566dd1e483674516d46d')
        self.assertEqual(self.comment.created_at, datetime(2012, 5, 22, 18, 40, 18, tzinfo=timezone.utc))
        self.assertEqual(self.comment.html_url, 'https://github.com/jacquev6/PyGithub/commit/6945921c529be14c3a8f566dd1e483674516d46d#commitcomment-1361949')
        self.assertEqual(self.comment.id, 1361949)
        self.assertEqual(self.comment.line, None)
        self.assertEqual(self.comment.path, None)
        self.assertEqual(self.comment.position, None)
        self.assertEqual(self.comment.updated_at, datetime(2012, 5, 22, 18, 40, 18, tzinfo=timezone.utc))
        self.assertEqual(self.comment.url, 'https://api.github.com/repos/jacquev6/PyGithub/comments/1361949')
        self.assertEqual(self.comment.user.login, 'jacquev6')
        self.assertEqual(repr(self.comment), 'CommitComment(user=NamedUser(login="jacquev6"), id=1361949)')

    def testEdit(self):
        if False:
            for i in range(10):
                print('nop')
        self.comment.edit('Comment edited by PyGithub')

    def testDelete(self):
        if False:
            return 10
        self.comment.delete()

    def testGetReactions(self):
        if False:
            return 10
        reactions = self.comment.get_reactions()
        self.assertEqual(reactions[0].content, '+1')

    def testCreateReaction(self):
        if False:
            while True:
                i = 10
        reaction = self.comment.create_reaction('hooray')
        self.assertEqual(reaction.id, 17283092)
        self.assertEqual(reaction.content, 'hooray')

    def testDeleteReaction(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.comment.delete_reaction(85737646))