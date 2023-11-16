from django.test import TestCase
from readthedocs.projects.models import Project

class TestRepoParsing(TestCase):
    fixtures = ['eric', 'test_data']

    def setUp(self):
        if False:
            print('Hello World!')
        self.client.login(username='eric', password='test')
        self.pip = Project.objects.get(slug='pip')
        self.version = self.pip.versions.create_latest()

    def test_github(self):
        if False:
            return 10
        self.pip.repo = 'https://github.com/user/repo'
        self.assertEqual(self.version.get_github_url(docroot='/docs/', filename='file'), 'https://github.com/user/repo/blob/master/docs/file.rst')
        self.pip.repo = 'https://github.com/user/repo/'
        self.assertEqual(self.version.get_github_url(docroot='/docs/', filename='file'), 'https://github.com/user/repo/blob/master/docs/file.rst')
        self.pip.repo = 'https://github.com/user/repo.github.io'
        self.assertEqual(self.version.get_github_url(docroot='/docs/', filename='file'), 'https://github.com/user/repo.github.io/blob/master/docs/file.rst')
        self.pip.repo = 'https://github.com/user/repo.github.io/'
        self.assertEqual(self.version.get_github_url(docroot='/docs/', filename='file'), 'https://github.com/user/repo.github.io/blob/master/docs/file.rst')
        self.pip.repo = 'https://github.com/user/repo.git'
        self.assertEqual(self.version.get_github_url(docroot='/docs/', filename='file'), 'https://github.com/user/repo/blob/master/docs/file.rst')
        self.pip.repo = 'https://github.com/user/repo.github.io.git'
        self.assertEqual(self.version.get_github_url(docroot='/docs/', filename='file'), 'https://github.com/user/repo.github.io/blob/master/docs/file.rst')
        self.pip.repo = 'https://github.com/user/repo.git.git'
        self.assertEqual(self.version.get_github_url(docroot='/docs/', filename='file'), 'https://github.com/user/repo.git/blob/master/docs/file.rst')
        self.pip.repo = 'https://github.com/user/repo/'
        self.assertEqual(self.version.get_github_url(docroot='/docs/', filename=''), 'https://github.com/user/repo/blob/master/docs/')

    def test_github_ssh(self):
        if False:
            for i in range(10):
                print('nop')
        self.pip.repo = 'git@github.com:user/repo.git'
        self.assertEqual(self.version.get_github_url(docroot='/docs/', filename='file'), 'https://github.com/user/repo/blob/master/docs/file.rst')
        self.pip.repo = 'git@github.com:user/repo.github.io.git'
        self.assertEqual(self.version.get_github_url(docroot='/docs/', filename='file'), 'https://github.com/user/repo.github.io/blob/master/docs/file.rst')

    def test_gitlab(self):
        if False:
            for i in range(10):
                print('nop')
        self.pip.repo = 'https://gitlab.com/user/repo'
        self.assertEqual(self.version.get_gitlab_url(docroot='/foo/bar/', filename='file'), 'https://gitlab.com/user/repo/blob/master/foo/bar/file.rst')
        self.pip.repo = 'https://gitlab.com/user/repo/'
        self.assertEqual(self.version.get_gitlab_url(docroot='/foo/bar/', filename='file'), 'https://gitlab.com/user/repo/blob/master/foo/bar/file.rst')
        self.pip.repo = 'https://gitlab.com/user/repo.gitlab.io'
        self.assertEqual(self.version.get_gitlab_url(docroot='/foo/bar/', filename='file'), 'https://gitlab.com/user/repo.gitlab.io/blob/master/foo/bar/file.rst')
        self.pip.repo = 'https://gitlab.com/user/repo.gitlab.io/'
        self.assertEqual(self.version.get_gitlab_url(docroot='/foo/bar/', filename='file'), 'https://gitlab.com/user/repo.gitlab.io/blob/master/foo/bar/file.rst')
        self.pip.repo = 'https://gitlab.com/user/repo.git'
        self.assertEqual(self.version.get_gitlab_url(docroot='/foo/bar/', filename='file'), 'https://gitlab.com/user/repo/blob/master/foo/bar/file.rst')
        self.pip.repo = 'https://gitlab.com/user/repo.gitlab.io.git'
        self.assertEqual(self.version.get_gitlab_url(docroot='/foo/bar/', filename='file'), 'https://gitlab.com/user/repo.gitlab.io/blob/master/foo/bar/file.rst')
        self.pip.repo = 'https://gitlab.com/user/repo.git.git'
        self.assertEqual(self.version.get_gitlab_url(docroot='/foo/bar/', filename='file'), 'https://gitlab.com/user/repo.git/blob/master/foo/bar/file.rst')
        self.pip.repo = 'https://gitlab.com/user/repo.git'
        self.assertEqual(self.version.get_gitlab_url(docroot='/foo/bar/', filename=''), 'https://gitlab.com/user/repo/blob/master/foo/bar/')

    def test_gitlab_ssh(self):
        if False:
            for i in range(10):
                print('nop')
        self.pip.repo = 'git@gitlab.com:user/repo.git'
        self.assertEqual(self.version.get_gitlab_url(docroot='/foo/bar/', filename='file'), 'https://gitlab.com/user/repo/blob/master/foo/bar/file.rst')
        self.pip.repo = 'git@gitlab.com:user/repo.gitlab.io.git'
        self.assertEqual(self.version.get_gitlab_url(docroot='/foo/bar/', filename='file'), 'https://gitlab.com/user/repo.gitlab.io/blob/master/foo/bar/file.rst')

    def test_bitbucket(self):
        if False:
            i = 10
            return i + 15
        self.pip.repo = 'https://bitbucket.org/user/repo'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename='file'), 'https://bitbucket.org/user/repo/src/master/foo/bar/file.rst')
        self.pip.repo = 'https://bitbucket.org/user/repo/'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename='file'), 'https://bitbucket.org/user/repo/src/master/foo/bar/file.rst')
        self.pip.repo = 'https://bitbucket.org/user/repo.gitbucket.io'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename='file'), 'https://bitbucket.org/user/repo.gitbucket.io/src/master/foo/bar/file.rst')
        self.pip.repo = 'https://bitbucket.org/user/repo.gitbucket.io/'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename='file'), 'https://bitbucket.org/user/repo.gitbucket.io/src/master/foo/bar/file.rst')
        self.pip.repo = 'https://bitbucket.org/user/repo.git'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename='file'), 'https://bitbucket.org/user/repo/src/master/foo/bar/file.rst')
        self.pip.repo = 'https://bitbucket.org/user/repo.gitbucket.io.git'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename='file'), 'https://bitbucket.org/user/repo.gitbucket.io/src/master/foo/bar/file.rst')
        self.pip.repo = 'https://bitbucket.org/user/repo.git.git'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename='file'), 'https://bitbucket.org/user/repo.git/src/master/foo/bar/file.rst')
        self.pip.repo = 'https://bitbucket.org/user/repo/'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename=''), 'https://bitbucket.org/user/repo/src/master/foo/bar/')

    def test_bitbucket_https(self):
        if False:
            print('Hello World!')
        self.pip.repo = 'https://user@bitbucket.org/user/repo.git'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename='file'), 'https://bitbucket.org/user/repo/src/master/foo/bar/file.rst')
        self.pip.repo = 'https://user@bitbucket.org/user/repo.gitbucket.io.git'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename='file'), 'https://bitbucket.org/user/repo.gitbucket.io/src/master/foo/bar/file.rst')

    def test_bitbucket_ssh(self):
        if False:
            while True:
                i = 10
        self.pip.repo = 'git@bitbucket.org:user/repo.git'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename='file'), 'https://bitbucket.org/user/repo/src/master/foo/bar/file.rst')
        self.pip.repo = 'git@bitbucket.org:user/repo.gitbucket.io.git'
        self.assertEqual(self.version.get_bitbucket_url(docroot='/foo/bar/', filename='file'), 'https://bitbucket.org/user/repo.gitbucket.io/src/master/foo/bar/file.rst')