from . import Framework

class GitRef(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.ref = self.g.get_user().get_repo('PyGithub').get_git_ref('heads/BranchCreatedByPyGithub')

    def testAttributes(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.ref.object.sha, '1292bf0e22c796e91cc3d6e24b544aece8c21f2a')
        self.assertEqual(self.ref.object.type, 'commit')
        self.assertEqual(self.ref.object.url, 'https://api.github.com/repos/jacquev6/PyGithub/git/commits/1292bf0e22c796e91cc3d6e24b544aece8c21f2a')
        self.assertEqual(self.ref.ref, 'refs/heads/BranchCreatedByPyGithub')
        self.assertEqual(self.ref.url, 'https://api.github.com/repos/jacquev6/PyGithub/git/refs/heads/BranchCreatedByPyGithub')
        self.assertEqual(repr(self.ref), 'GitRef(ref="refs/heads/BranchCreatedByPyGithub")')
        self.assertEqual(repr(self.ref.object), 'GitObject(sha="1292bf0e22c796e91cc3d6e24b544aece8c21f2a")')

    def testEdit(self):
        if False:
            i = 10
            return i + 15
        self.ref.edit('04cde900a0775b51f762735637bd30de392a2793')

    def testEditWithForce(self):
        if False:
            i = 10
            return i + 15
        self.ref.edit('4303c5b90e2216d927155e9609436ccb8984c495', force=True)

    def testDelete(self):
        if False:
            return 10
        self.ref.delete()