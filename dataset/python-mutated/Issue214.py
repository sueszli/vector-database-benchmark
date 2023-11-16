from . import Framework

class Issue214(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.repo = self.g.get_user().get_repo('PyGithub')
        self.issue = self.repo.get_issue(1)

    def testAssignees(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.repo.has_in_assignees('farrd'))
        self.assertFalse(self.repo.has_in_assignees('fake'))

    def testCollaborators(self):
        if False:
            return 10
        self.assertTrue(self.repo.has_in_collaborators('farrd'))
        self.assertFalse(self.repo.has_in_collaborators('fake'))
        self.assertFalse(self.repo.has_in_collaborators('marcmenges'))
        self.repo.add_to_collaborators('marcmenges')
        self.assertTrue(self.repo.has_in_collaborators('marcmenges'))
        self.repo.remove_from_collaborators('marcmenges')
        self.assertFalse(self.repo.has_in_collaborators('marcmenges'))

    def testEditIssue(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.issue.assignee, None)
        self.issue.edit(assignee='farrd')
        self.assertEqual(self.issue.assignee.login, 'farrd')
        self.issue.edit(assignee=None)
        self.assertEqual(self.issue.assignee, None)

    def testCreateIssue(self):
        if False:
            print('Hello World!')
        issue = self.repo.create_issue('Issue created by PyGithub', assignee='farrd')
        self.assertEqual(issue.assignee.login, 'farrd')

    def testGetIssues(self):
        if False:
            for i in range(10):
                print('nop')
        issues = self.repo.get_issues(assignee='farrd')
        for issue in issues:
            self.assertEqual(issue.assignee.login, 'farrd')