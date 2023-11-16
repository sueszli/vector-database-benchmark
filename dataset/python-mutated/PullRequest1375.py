from . import Framework

class PullRequest1375(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.pr = self.g.get_repo('rsn491/PyGithub').get_pulls()[0]

    def testCreateReviewCommentReply(self):
        if False:
            return 10
        comment_id = 373866377
        first_reply_body = 'Comment reply created by PyGithub'
        second_reply_body = 'Second comment reply created by PyGithub'
        first_reply = self.pr.create_review_comment_reply(comment_id, first_reply_body)
        second_reply = self.pr.create_review_comment_reply(first_reply.id, second_reply_body)
        self.assertEqual(first_reply.in_reply_to_id, comment_id)
        self.assertEqual(second_reply.in_reply_to_id, comment_id)
        self.assertEqual(first_reply.body, first_reply_body)
        self.assertEqual(second_reply.body, second_reply_body)