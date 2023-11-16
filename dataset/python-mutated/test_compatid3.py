from mutagen import id3
from test.picardtestcase import PicardTestCase
from picard.formats.id3 import Id3Encoding
from picard.formats.mutagenext import compatid3

class UpdateToV23Test(PicardTestCase):

    def test_keep_some_v24_tag(self):
        if False:
            return 10
        tags = compatid3.CompatID3()
        tags.add(id3.TSOP(encoding=Id3Encoding.LATIN1, text=['foo']))
        tags.add(id3.TSOA(encoding=Id3Encoding.LATIN1, text=['foo']))
        tags.add(id3.TSOT(encoding=Id3Encoding.LATIN1, text=['foo']))
        tags.update_to_v23()
        self.assertEqual(tags['TSOP'].text, ['foo'])
        self.assertEqual(tags['TSOA'].text, ['foo'])
        self.assertEqual(tags['TSOT'].text, ['foo'])

    def test_tdrc(self):
        if False:
            for i in range(10):
                print('nop')
        tags = compatid3.CompatID3()
        tags.add(id3.TDRC(encoding=Id3Encoding.UTF16, text='2003-04-05 12:03'))
        tags.update_to_v23()
        self.assertEqual(tags['TYER'].text, ['2003'])
        self.assertEqual(tags['TDAT'].text, ['0504'])
        self.assertEqual(tags['TIME'].text, ['1203'])

    def test_tdor(self):
        if False:
            for i in range(10):
                print('nop')
        tags = compatid3.CompatID3()
        tags.add(id3.TDOR(encoding=Id3Encoding.UTF16, text='2003-04-05 12:03'))
        tags.update_to_v23()
        self.assertEqual(tags['TORY'].text, ['2003'])

    def test_genre_from_v24_1(self):
        if False:
            while True:
                i = 10
        tags = compatid3.CompatID3()
        tags.add(id3.TCON(encoding=Id3Encoding.UTF16, text=['4', 'Rock']))
        tags.update_to_v23()
        self.assertEqual(tags['TCON'].text, ['Disco', 'Rock'])

    def test_genre_from_v24_2(self):
        if False:
            while True:
                i = 10
        tags = compatid3.CompatID3()
        tags.add(id3.TCON(encoding=Id3Encoding.UTF16, text=['RX', '3', 'CR']))
        tags.update_to_v23()
        self.assertEqual(tags['TCON'].text, ['Remix', 'Dance', 'Cover'])

    def test_genre_from_v23_1(self):
        if False:
            i = 10
            return i + 15
        tags = compatid3.CompatID3()
        tags.add(id3.TCON(encoding=Id3Encoding.UTF16, text=['(4)Rock']))
        tags.update_to_v23()
        self.assertEqual(tags['TCON'].text, ['Disco', 'Rock'])

    def test_genre_from_v23_2(self):
        if False:
            i = 10
            return i + 15
        tags = compatid3.CompatID3()
        tags.add(id3.TCON(encoding=Id3Encoding.UTF16, text=['(RX)(3)(CR)']))
        tags.update_to_v23()
        self.assertEqual(tags['TCON'].text, ['Remix', 'Dance', 'Cover'])