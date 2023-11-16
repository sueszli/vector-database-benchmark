from datetime import datetime, timezone
from . import Framework

class Issue54(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.repo = self.g.get_user().get_repo('TestRepo')

    def testConversion(self):
        if False:
            for i in range(10):
                print('nop')
        commit = self.repo.get_git_commit('73f320ae06cd565cf38faca34b6a482addfc721b')
        self.assertEqual(commit.message, 'Test commit created around Fri, 13 Jul 2012 18:43:21 GMT, that is vendredi 13 juillet 2012 20:43:21 GMT+2\n')
        self.assertEqual(commit.author.date, datetime(2012, 7, 13, 18, 47, 10, tzinfo=timezone.utc))