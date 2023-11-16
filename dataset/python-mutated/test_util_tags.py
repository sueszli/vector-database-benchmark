from test.picardtestcase import PicardTestCase
from picard.util.tags import display_tag_name, parse_comment_tag

class UtilTagsTest(PicardTestCase):

    def test_display_tag_name(self):
        if False:
            print('Hello World!')
        self.assertEqual('Artist', display_tag_name('artist'))
        self.assertEqual('Lyrics', display_tag_name('lyrics:'))
        self.assertEqual('Comment [Foo]', display_tag_name('comment:Foo'))

    def test_parse_comment_tag(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(('XXX', 'foo'), parse_comment_tag('comment:XXX:foo'))
        self.assertEqual(('eng', 'foo'), parse_comment_tag('comment:foo'))
        self.assertEqual(('eng', ''), parse_comment_tag('comment'))