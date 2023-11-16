from . import Framework

class Search(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()

    def testSearchUsers(self):
        if False:
            for i in range(10):
                print('nop')
        users = self.g.search_users('vincent', sort='followers', order='desc')
        self.assertEqual(users.totalCount, 2781)

    def testPaginateSearchUsers(self):
        if False:
            for i in range(10):
                print('nop')
        users = self.g.search_users('', location='Berlin')
        self.assertListKeyBegin(users, lambda u: u.login, ['cloudhead', 'felixge', 'sferik', 'rkh', 'jezdez', 'janl', 'marijnh', 'nikic', 'igorw', 'froschi', 'svenfuchs', 'omz', 'chad', 'bergie', 'roidrage', 'pcalcado', 'durran', 'hukl', 'mttkay', 'aFarkas', 'ole', 'hagenburger', 'jberkel', 'naderman', 'joshk', 'pudo', 'robb', 'josephwilk', 'hanshuebner', 'txus', 'paulasmuth', 'splitbrain', 'langalex', 'bendiken', 'stefanw'])
        self.assertEqual(users.totalCount, 6038)

    def testGetPageOnSearchUsers(self):
        if False:
            for i in range(10):
                print('nop')
        users = self.g.search_users('', location='Berlin')
        self.assertEqual([u.login for u in users.get_page(7)], ['ursachec', 'bitboxer', 'fs111', 'michenriksen', 'witsch', 'booo', 'mortice', 'r0man', 'MikeBild', 'mhagger', 'bkw', 'fwbrasil', 'mschneider', 'lydiapintscher', 'asksven', 'iamtimm', 'sneak', 'kr1sp1n', 'Feh', 'GordonLesti', 'annismckenzie', 'eskimoblood', 'tsujigiri', 'riethmayer', 'lauritzthamsen', 'scotchi', 'peritor', 'toto', 'hwaxxer', 'lukaszklis'])

    def testSearchRepos(self):
        if False:
            return 10
        repos = self.g.search_repositories('github', sort='stars', order='desc', language='Python')
        self.assertListKeyBegin(repos, lambda r: r.full_name, ['kennethreitz/legit', 'RuudBurger/CouchPotatoV1', 'gelstudios/gitfiti', 'gpjt/webgl-lessons', 'jacquev6/PyGithub', 'aaasen/github_globe', 'hmason/gitmarks', 'dnerdy/factory_boy', 'binaryage/drydrop', 'bgreenlee/sublime-github', 'karan/HackerNewsAPI', 'mfenniak/pyPdf', 'skazhy/github-decorator', 'llvmpy/llvmpy', 'lexrupy/gmate', 'ask/python-github2', 'audreyr/cookiecutter-pypackage', 'tabo/django-treebeard', 'dbr/tvdb_api', 'jchris/couchapp', 'joeyespo/grip', 'nigelsmall/py2neo', 'ask/chishop', 'sigmavirus24/github3.py', 'jsmits/github-cli', 'lincolnloop/django-layout', 'amccloud/django-project-skel', 'Stiivi/brewery', 'webpy/webpy.github.com', 'dustin/py-github', 'logsol/Github-Auto-Deploy', 'cloudkick/libcloud', 'berkerpeksag/github-badge', 'bitprophet/ssh', 'azavea/OpenTreeMap'])

    def testSearchReposWithNoResults(self):
        if False:
            i = 10
            return i + 15
        repos = self.g.search_repositories('doesnotexist')
        self.assertEqual(repos.totalCount, 0)

    def testSearchIssues(self):
        if False:
            print('Hello World!')
        issues = self.g.search_issues('compile', sort='comments', order='desc', language='C++')
        self.assertListKeyBegin(issues, lambda i: i.id, [12068673, 23250111, 14371957, 9423897, 24277400, 2408877, 11338741, 13980502, 27697165, 23102422])

    def testPaginateSearchCommits(self):
        if False:
            while True:
                i = 10
        commits = self.g.search_commits(query='hash:5b0224e868cc9242c9450ef02efbe3097abd7ba2')
        self.assertEqual(commits.totalCount, 3)

    def testSearchCommits(self):
        if False:
            return 10
        commits = self.g.search_commits(query='hash:1265747e992ba7d34a469b6b2f527809f8bf7067', sort='author-date', order='asc', merge='false')
        self.assertEqual(commits.totalCount, 2)

    def testSearchTopics(self):
        if False:
            return 10
        topics = self.g.search_topics('python', repositories='>950')
        self.assertListKeyBegin(topics, lambda r: r.name, ['python', 'django', 'flask', 'ruby', 'scikit-learn', 'wagtail'])

    def testPaginateSearchTopics(self):
        if False:
            return 10
        repos = self.g.search_topics('python', repositories='>950')
        self.assertEqual(repos.totalCount, 6)

    def testSearchCode(self):
        if False:
            print('Hello World!')
        files = self.g.search_code('toto', sort='indexed', order='asc', user='jacquev6')
        self.assertListKeyEqual(files, lambda f: f.name, ['Commit.setUp.txt', 'PullRequest.testGetFiles.txt', 'NamedUser.testGetEvents.txt', 'PullRequest.testCreateComment.txt', 'PullRequestFile.setUp.txt', 'Repository.testGetIssuesWithWildcards.txt', 'Repository.testGetIssuesWithArguments.txt', 'test_ebnf.cpp', 'test_abnf.cpp', 'PullRequestFile.py', 'SystemCalls.py', 'tests.py', 'LexerTestCase.py', 'ParserTestCase.py'])
        self.assertEqual(files[0].repository.full_name, 'jacquev6/PyGithub')
        content = files[0].decoded_content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        self.assertEqual(content[:30], 'https\nGET\napi.github.com\nNone\n')

    def testSearchHighlightingCode(self):
        if False:
            return 10
        files = self.g.search_code('toto', sort='indexed', order='asc', user='jacquev6', highlight=True)
        self.assertTrue(files[0].text_matches)

    def testUrlquotingOfQualifiers(self):
        if False:
            return 10
        issues = self.g.search_issues('repo:saltstack/salt-api type:Issues', updated='>2014-03-04T18:28:11Z')
        self.assertEqual(issues[0].id, 29138794)

    def testUrlquotingOfQuery(self):
        if False:
            return 10
        issues = self.g.search_issues('repo:saltstack/salt-api type:Issues updated:>2014-03-04T18:28:11Z')
        self.assertEqual(issues[0].id, 29138794)