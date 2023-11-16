from datetime import datetime, timezone
from . import Framework

class GitTag(Framework.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.tag = self.g.get_user().get_repo('PyGithub').get_git_tag('f5f37322407b02a80de4526ad88d5f188977bc3c')

    def testAttributes(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.tag.message, 'Version 0.6\n')
        self.assertEqual(self.tag.object.sha, '4303c5b90e2216d927155e9609436ccb8984c495')
        self.assertEqual(self.tag.object.type, 'commit')
        self.assertEqual(self.tag.object.url, 'https://api.github.com/repos/jacquev6/PyGithub/git/commits/4303c5b90e2216d927155e9609436ccb8984c495')
        self.assertEqual(self.tag.sha, 'f5f37322407b02a80de4526ad88d5f188977bc3c')
        self.assertEqual(self.tag.tag, 'v0.6')
        self.assertEqual(self.tag.tagger.date, datetime(2012, 5, 10, 18, 14, 15, tzinfo=timezone.utc))
        self.assertEqual(self.tag.tagger.email, 'vincent@vincent-jacques.net')
        self.assertEqual(self.tag.tagger.name, 'Vincent Jacques')
        self.assertEqual(self.tag.url, 'https://api.github.com/repos/jacquev6/PyGithub/git/tags/f5f37322407b02a80de4526ad88d5f188977bc3c')
        self.assertEqual(repr(self.tag), 'GitTag(tag="v0.6", sha="f5f37322407b02a80de4526ad88d5f188977bc3c")')