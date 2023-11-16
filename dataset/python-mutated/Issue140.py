from . import Framework

class Issue140(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.repo = self.g.get_repo('twitter/bootstrap')

    def testGetDirContentsThenLazyCompletionOfFile(self):
        if False:
            return 10
        contents = self.repo.get_contents('js')
        self.assertEqual(len(contents), 15)
        n = 0
        for content in contents:
            if content.path == 'js/bootstrap-affix.js':
                self.assertEqual(len(content.content), 4722)
                n += 1
            elif content.path == 'js/tests':
                self.assertEqual(content.content, None)
                n += 1
        self.assertEqual(n, 2)

    def testGetFileContents(self):
        if False:
            print('Hello World!')
        contents = self.repo.get_contents('js/bootstrap-affix.js')
        self.assertEqual(contents.encoding, 'base64')
        self.assertEqual(contents.url, 'https://api.github.com/repos/twitter/bootstrap/contents/js/bootstrap-affix.js')
        self.assertEqual(len(contents.content), 4722)

    def testGetDirContentsWithRef(self):
        if False:
            return 10
        self.assertEqual(len(self.repo.get_contents('js', '8c7f9c66a7d12f47f50618ef420868fe836d0c33')), 15)