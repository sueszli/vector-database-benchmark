from . import Framework

class Label(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.label = self.g.get_user().get_repo('PyGithub').get_label('Bug')

    def testAttributes(self):
        if False:
            return 10
        self.assertEqual(self.label.color, 'e10c02')
        self.assertEqual(self.label.name, 'Bug')
        self.assertIsNone(self.label.description)
        self.assertEqual(self.label.url, 'https://api.github.com/repos/jacquev6/PyGithub/labels/Bug')
        self.assertEqual(repr(self.label), 'Label(name="Bug")')

    def testEdit(self):
        if False:
            print('Hello World!')
        self.label.edit('LabelEditedByPyGithub', '0000ff', 'Description of LabelEditedByPyGithub')
        self.assertEqual(self.label.color, '0000ff')
        self.assertEqual(self.label.description, 'Description of LabelEditedByPyGithub')
        self.assertEqual(self.label.name, 'LabelEditedByPyGithub')
        self.assertEqual(self.label.url, 'https://api.github.com/repos/jacquev6/PyGithub/labels/LabelEditedByPyGithub')

    def testDelete(self):
        if False:
            while True:
                i = 10
        self.label.delete()