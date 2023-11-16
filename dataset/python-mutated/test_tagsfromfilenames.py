from test.picardtestcase import PicardTestCase
from picard.ui.tagsfromfilenames import TagMatchExpression

class TagMatchExpressionTest(PicardTestCase):

    def test_parse_tags(self):
        if False:
            return 10
        expression = TagMatchExpression('%tracknumber% - %title%')
        expected_tags = ['tracknumber', 'title']
        self.assertEqual(expected_tags, expression.matched_tags)
        files = ['042 - The Title', '042 - The Title.mp3', '/foo/042 - The Title/foo/042 - The Title.mp3/042 - The Title/042 - The Title.mp3C:\\foo\\042 - The Title.mp3']
        for filename in files:
            matches = expression.match_file(filename)
            self.assertEqual(['42'], matches['tracknumber'])
            self.assertEqual(['The Title'], matches['title'])

    def test_parse_tags_with_path(self):
        if False:
            for i in range(10):
                print('nop')
        expression = TagMatchExpression('%artist%/%album%/%tracknumber% - %title%')
        expected_tags = ['artist', 'album', 'tracknumber', 'title']
        self.assertEqual(expected_tags, expression.matched_tags)
        files = ['The Artist/The Album/01 - The Title', 'The Artist/The Album/01 - The Title.wv', 'C:\\foo\\The Artist\\The Album\\01 - The Title.wv']
        for filename in files:
            matches = expression.match_file(filename)
            self.assertEqual(['The Artist'], matches['artist'])
            self.assertEqual(['The Album'], matches['album'])
            self.assertEqual(['1'], matches['tracknumber'])
            self.assertEqual(['The Title'], matches['title'])

    def test_parse_replace_underscores(self):
        if False:
            print('Hello World!')
        expression = TagMatchExpression('%artist%-%title%', replace_underscores=True)
        matches = expression.match_file('Some_Artist-Some_Title.ogg')
        self.assertEqual(['Some Artist'], matches['artist'])
        self.assertEqual(['Some Title'], matches['title'])

    def test_parse_tags_duplicates(self):
        if False:
            return 10
        expression = TagMatchExpression('%dummy% %title% %dummy%')
        expected_tags = ['dummy', 'title']
        self.assertEqual(expected_tags, expression.matched_tags)
        matches = expression.match_file('foo title bar')
        self.assertEqual(['title'], matches['title'])
        self.assertEqual(['foo', 'bar'], matches['dummy'])

    def test_parse_tags_hidden(self):
        if False:
            for i in range(10):
                print('nop')
        expression = TagMatchExpression('%_dummy% %title% %_dummy%')
        expected_tags = ['~dummy', 'title']
        self.assertEqual(expected_tags, expression.matched_tags)
        matches = expression.match_file('foo title bar')
        self.assertEqual(['title'], matches['title'])
        self.assertEqual(['foo', 'bar'], matches['~dummy'])

    def test_parse_empty(self):
        if False:
            for i in range(10):
                print('nop')
        expression = TagMatchExpression('')
        expected_tags = []
        self.assertEqual(expected_tags, expression.matched_tags)