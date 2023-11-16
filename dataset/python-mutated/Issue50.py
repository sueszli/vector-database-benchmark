from . import Framework

class Issue50(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.repo = self.g.get_user().get_repo('PyGithub')
        self.issue = self.repo.get_issue(50)
        self.labelName = 'Label with spaces and strange characters (&*#$)'

    def testCreateLabel(self):
        if False:
            i = 10
            return i + 15
        label = self.repo.create_label(self.labelName, 'ffff00')
        self.assertEqual(label.name, self.labelName)

    def testGetLabel(self):
        if False:
            while True:
                i = 10
        label = self.repo.get_label(self.labelName)
        self.assertEqual(label.name, self.labelName)

    def testGetLabels(self):
        if False:
            return 10
        self.assertListKeyEqual(self.repo.get_labels(), lambda l: l.name, ['Refactoring', 'Public interface', 'Functionalities', 'Project management', 'Bug', 'Question', 'RequestedByUser', self.labelName])

    def testAddLabelToIssue(self):
        if False:
            for i in range(10):
                print('nop')
        self.issue.add_to_labels(self.repo.get_label(self.labelName))

    def testRemoveLabelFromIssue(self):
        if False:
            print('Hello World!')
        self.issue.remove_from_labels(self.repo.get_label(self.labelName))

    def testSetIssueLabels(self):
        if False:
            while True:
                i = 10
        self.issue.set_labels(self.repo.get_label('Bug'), self.repo.get_label('RequestedByUser'), self.repo.get_label(self.labelName))

    def testIssueLabels(self):
        if False:
            i = 10
            return i + 15
        self.assertListKeyEqual(self.issue.labels, lambda l: l.name, ['Bug', self.labelName, 'RequestedByUser'])

    def testIssueGetLabels(self):
        if False:
            return 10
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Bug', self.labelName, 'RequestedByUser'])

    def testGetIssuesWithLabel(self):
        if False:
            print('Hello World!')
        self.assertListKeyEqual(self.repo.get_issues(labels=[self.repo.get_label(self.labelName)]), lambda i: i.number, [52, 50])

    def testCreateIssueWithLabel(self):
        if False:
            while True:
                i = 10
        issue = self.repo.create_issue('Issue created by PyGithub to test issue #50', labels=[self.repo.get_label(self.labelName)])
        self.assertListKeyEqual(issue.labels, lambda l: l.name, [self.labelName])
        self.assertEqual(issue.number, 52)