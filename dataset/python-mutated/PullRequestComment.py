from datetime import datetime, timezone
from . import Framework

class PullRequestComment(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.comment = self.g.get_user().get_repo('PyGithub').get_pull(31).get_comment(886298)

    def testAttributes(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.comment.body, 'Comment created by PyGithub')
        self.assertEqual(self.comment.commit_id, '8a4f306d4b223682dd19410d4a9150636ebe4206')
        self.assertEqual(self.comment.created_at, datetime(2012, 5, 27, 9, 40, 12, tzinfo=timezone.utc))
        self.assertEqual(self.comment.id, 886298)
        self.assertEqual(self.comment.original_commit_id, '8a4f306d4b223682dd19410d4a9150636ebe4206')
        self.assertEqual(self.comment.original_position, 5)
        self.assertEqual(self.comment.path, 'src/github/Issue.py')
        self.assertEqual(self.comment.position, 5)
        self.assertEqual(self.comment.updated_at, datetime(2012, 5, 27, 9, 40, 12, tzinfo=timezone.utc))
        self.assertEqual(self.comment.url, 'https://api.github.com/repos/jacquev6/PyGithub/pulls/comments/886298')
        self.assertEqual(self.comment.user.login, 'jacquev6')
        self.assertEqual(self.comment.html_url, 'https://github.com/jacquev6/PyGithub/pull/170#issuecomment-18637907')
        self.assertEqual(repr(self.comment), 'PullRequestComment(user=NamedUser(login="jacquev6"), id=886298)')

    def testEdit(self):
        if False:
            return 10
        self.comment.edit('Comment edited by PyGithub')
        self.assertEqual(self.comment.body, 'Comment edited by PyGithub')

    def testDelete(self):
        if False:
            print('Hello World!')
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
        self.assertEqual(reaction.id, 17283822)
        self.assertEqual(reaction.content, 'hooray')

    def testDeleteReaction(self):
        if False:
            return 10
        self.assertTrue(self.comment.delete_reaction(85750463))