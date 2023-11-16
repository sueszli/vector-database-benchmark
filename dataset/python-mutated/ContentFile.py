from . import Framework

class ContentFile(Framework.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.file = self.g.get_user().get_repo('PyGithub').get_readme()

    def testAttributes(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.file.type, 'file')
        self.assertEqual(self.file.encoding, 'base64')
        self.assertEqual(self.file.size, 7531)
        self.assertEqual(self.file.name, 'ReadMe.md')
        self.assertEqual(self.file.path, 'ReadMe.md')
        self.assertEqual(len(self.file.content), 10212)
        self.assertEqual(len(self.file.decoded_content), 7531)
        self.assertEqual(self.file.sha, '5628799a7d517a4aaa0c1a7004d07569cd154df0')
        self.assertEqual(self.file.download_url, 'https://raw.githubusercontent.com/jacquev6/PyGithub/master/README.md')
        self.assertIsNone(self.file.license)
        self.assertEqual(repr(self.file), 'ContentFile(path="ReadMe.md")')