from test.picardtestcase import PicardTestCase
from picard.formats import mutagenext

class MutagenExtTest(PicardTestCase):

    def test_delall_ci(self):
        if False:
            while True:
                i = 10
        tags = {'TAGNAME:ABC': 'a', 'tagname:abc': 'a', 'TagName:Abc': 'a', 'OtherTag': 'a'}
        mutagenext.delall_ci(tags, 'tagname:Abc')
        self.assertEqual({'OtherTag': 'a'}, tags)