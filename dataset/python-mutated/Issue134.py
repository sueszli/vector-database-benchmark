import github
from . import Framework

class Issue134(Framework.BasicTestCase):

    def testGetAuthorizationsFailsWhenAutenticatedThroughOAuth(self):
        if False:
            print('Hello World!')
        g = github.Github(auth=self.oauth_token)
        with self.assertRaises(github.GithubException) as raisedexp:
            list(g.get_user().get_authorizations())
        self.assertEqual(raisedexp.exception.status, 404)

    def testGetAuthorizationsSucceedsWhenAutenticatedThroughLoginPassword(self):
        if False:
            return 10
        g = github.Github(auth=self.login)
        self.assertListKeyEqual(g.get_user().get_authorizations(), lambda a: a.note, [None, None, 'cligh', None, None, 'GitHub Android App'])

    def testGetOAuthScopesFromHeader(self):
        if False:
            print('Hello World!')
        g = github.Github(auth=self.oauth_token)
        self.assertEqual(g.oauth_scopes, None)
        g.get_user().name
        self.assertEqual(g.oauth_scopes, ['repo', 'user', 'gist'])