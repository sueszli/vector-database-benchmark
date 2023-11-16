from datetime import datetime, timezone
from . import Framework

class IssueComment(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.comment = self.g.get_user().get_repo('PyGithub').get_issue(28).get_comment(5808311)

    def testAttributes(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.comment.body, 'Comment created by PyGithub')
        self.assertEqual(self.comment.created_at, datetime(2012, 5, 20, 11, 46, 42, tzinfo=timezone.utc))
        self.assertEqual(self.comment.id, 5808311)
        self.assertEqual(self.comment.updated_at, datetime(2012, 5, 20, 11, 46, 42, tzinfo=timezone.utc))
        self.assertEqual(self.comment.url, 'https://api.github.com/repos/jacquev6/PyGithub/issues/comments/5808311')
        self.assertEqual(self.comment.user.login, 'jacquev6')
        self.assertEqual(self.comment.html_url, 'https://github.com/jacquev6/PyGithub/issues/28#issuecomment-5808311')
        self.assertEqual(repr(self.comment), 'IssueComment(user=NamedUser(login="jacquev6"), id=5808311)')
        self.assertEqual(self.comment.reactions, {'+1': 1, '-1': 0, 'confused': 0, 'eyes': 0, 'heart': 0, 'hooray': 1, 'laugh': 0, 'rocket': 0, 'total_count': 2, 'url': 'https://api.github.com/repos/jacquev6/PyGithub/issues/comments/5808311/reactions'})

    def testEdit(self):
        if False:
            for i in range(10):
                print('nop')
        self.comment.edit('Comment edited by PyGithub')
        self.assertEqual(self.comment.body, 'Comment edited by PyGithub')
        self.assertEqual(self.comment.updated_at, datetime(2012, 5, 20, 11, 53, 59, tzinfo=timezone.utc))

    def testDelete(self):
        if False:
            while True:
                i = 10
        self.comment.delete()

    def testGetReactions(self):
        if False:
            print('Hello World!')
        reactions = self.comment.get_reactions()
        self.assertEqual(reactions[0].content, '+1')

    def testCreateReaction(self):
        if False:
            i = 10
            return i + 15
        reaction = self.comment.create_reaction('hooray')
        self.assertEqual(reaction.id, 17282654)
        self.assertEqual(reaction.content, 'hooray')

    def testDeleteReaction(self):
        if False:
            return 10
        self.assertTrue(self.comment.delete_reaction(85743754))