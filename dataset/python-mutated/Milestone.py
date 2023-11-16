from datetime import date, datetime, timezone
from . import Framework

class Milestone(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.milestone = self.g.get_user().get_repo('PyGithub').get_milestone(1)

    def testAttributes(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.milestone.closed_issues, 2)
        self.assertEqual(self.milestone.created_at, datetime(2012, 3, 8, 12, 22, 10, tzinfo=timezone.utc))
        self.assertEqual(self.milestone.description, '')
        self.assertEqual(self.milestone.due_on, datetime(2012, 3, 13, 7, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(self.milestone.id, 93546)
        self.assertEqual(self.milestone.number, 1)
        self.assertEqual(self.milestone.open_issues, 0)
        self.assertEqual(self.milestone.state, 'closed')
        self.assertEqual(self.milestone.title, 'Version 0.4')
        self.assertEqual(self.milestone.url, 'https://api.github.com/repos/jacquev6/PyGithub/milestones/1')
        self.assertEqual(self.milestone.creator.login, 'jacquev6')
        self.assertEqual(repr(self.milestone), 'Milestone(title="Version 0.4", number=1)')

    def testEditWithMinimalParameters(self):
        if False:
            while True:
                i = 10
        self.milestone.edit('Title edited by PyGithub')
        self.assertEqual(self.milestone.title, 'Title edited by PyGithub')

    def testEditWithAllParameters(self):
        if False:
            return 10
        self.milestone.edit('Title edited twice by PyGithub', 'closed', 'Description edited by PyGithub', due_on=date(2012, 6, 16))
        self.assertEqual(self.milestone.title, 'Title edited twice by PyGithub')
        self.assertEqual(self.milestone.state, 'closed')
        self.assertEqual(self.milestone.description, 'Description edited by PyGithub')
        self.assertEqual(self.milestone.due_on, datetime(2012, 6, 16, 7, 0, 0, tzinfo=timezone.utc))

    def testGetLabels(self):
        if False:
            i = 10
            return i + 15
        self.assertListKeyEqual(self.milestone.get_labels(), lambda l: l.name, ['Public interface', 'Project management'])

    def testDelete(self):
        if False:
            i = 10
            return i + 15
        self.milestone.delete()