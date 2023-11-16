import github
from . import Framework

class Enterprise(Framework.BasicTestCase):

    def testHttps(self):
        if False:
            i = 10
            return i + 15
        g = github.Github(auth=self.login, base_url='https://my.enterprise.com')
        self.assertListKeyEqual(g.get_user().get_repos(), lambda r: r.name, ['TestPyGithub', 'django', 'PyGithub', 'developer.github.com', 'acme-public-website', 'C4Planner', 'Hacking', 'vincent-jacques.net', 'Contests', 'Candidates', 'Tests', 'DrawTurksHead', 'DrawSyntax', 'QuadProgMm', 'Boost.HierarchicalEnum', 'ViDE'])

    def testHttp(self):
        if False:
            print('Hello World!')
        g = github.Github(auth=self.login, base_url='http://my.enterprise.com')
        self.assertListKeyEqual(g.get_user().get_repos(), lambda r: r.name, ['TestPyGithub', 'django', 'PyGithub', 'developer.github.com', 'acme-public-website', 'C4Planner', 'Hacking', 'vincent-jacques.net', 'Contests', 'Candidates', 'Tests', 'DrawTurksHead', 'DrawSyntax', 'QuadProgMm', 'Boost.HierarchicalEnum', 'ViDE'])

    def testUnknownUrlScheme(self):
        if False:
            return 10
        with self.assertRaises(AssertionError) as raisedexp:
            github.Github(auth=self.login, base_url='foobar://my.enterprise.com')
        self.assertEqual(raisedexp.exception.args[0], 'Unknown URL scheme')

    def testLongUrl(self):
        if False:
            while True:
                i = 10
        g = github.Github(auth=self.login, base_url='http://my.enterprise.com/path/to/github')
        repos = g.get_user().get_repos()
        self.assertListKeyEqual(repos, lambda r: r.name, ['TestPyGithub', 'django', 'PyGithub', 'developer.github.com', 'acme-public-website', 'C4Planner', 'Hacking', 'vincent-jacques.net', 'Contests', 'Candidates', 'Tests', 'DrawTurksHead', 'DrawSyntax', 'QuadProgMm', 'Boost.HierarchicalEnum', 'ViDE'])
        self.assertEqual(repos[0].owner.name, 'Vincent Jacques')

    def testSpecificPort(self):
        if False:
            while True:
                i = 10
        g = github.Github(auth=self.login, base_url='http://my.enterprise.com:8080')
        self.assertListKeyEqual(g.get_user().get_repos(), lambda r: r.name, ['TestPyGithub', 'django', 'PyGithub', 'developer.github.com', 'acme-public-website', 'C4Planner', 'Hacking', 'vincent-jacques.net', 'Contests', 'Candidates', 'Tests', 'DrawTurksHead', 'DrawSyntax', 'QuadProgMm', 'Boost.HierarchicalEnum', 'ViDE'])