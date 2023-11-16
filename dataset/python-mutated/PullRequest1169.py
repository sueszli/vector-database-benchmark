from . import Framework

class PullRequest1169(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        ferada_repo = self.g.get_repo('coleslaw-org/coleslaw', lazy=True)
        self.pull = ferada_repo.get_pull(173)

    def testReviewApproveWithoutBody(self):
        if False:
            for i in range(10):
                print('nop')
        r = self.pull.create_review(event='APPROVE')
        self.assertEqual(r.id, 261942907)
        self.assertEqual(r.user.login, 'Ferada')