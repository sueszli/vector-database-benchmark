import github
from . import Framework

class Issue80(Framework.BasicTestCase):

    def testIgnoreHttpsFromGithubEnterprise(self):
        if False:
            i = 10
            return i + 15
        g = github.Github(auth=self.login, base_url='http://my.enterprise.com/some/prefix')
        org = g.get_organization('BeaverSoftware')
        self.assertEqual(org.url, 'https://my.enterprise.com/some/prefix/orgs/BeaverSoftware')
        self.assertListKeyEqual(org.get_repos(), lambda r: r.name, ['FatherBeaver', 'TestPyGithub'])

    def testIgnoreHttpsFromGithubEnterpriseWithPort(self):
        if False:
            return 10
        g = github.Github(auth=self.login, base_url='http://my.enterprise.com:1234/some/prefix')
        org = g.get_organization('BeaverSoftware')
        self.assertEqual(org.url, 'https://my.enterprise.com:1234/some/prefix/orgs/BeaverSoftware')
        self.assertListKeyEqual(org.get_repos(), lambda r: r.name, ['FatherBeaver', 'TestPyGithub'])