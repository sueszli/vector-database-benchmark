from tests import Framework

class Autolink(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        links = [x for x in self.g.get_user('theCapypara').get_repo('PyGithub').get_autolinks() if x.id == 209614]
        self.assertEqual(1, len(links), 'There must be exactly one autolink with the ID 209614.')
        self.link = links[0]

    def testAttributes(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.link.id, 209614)
        self.assertEqual(self.link.key_prefix, 'DUMMY-')
        self.assertEqual(self.link.url_template, 'https://github.com/PyGithub/PyGithub/issues/<num>')
        self.assertEqual(self.link.is_alphanumeric, True)