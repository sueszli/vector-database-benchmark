from test.picardtestcase import PicardTestCase
from picard.coverart.utils import image_type_as_id3_num, image_type_from_id3_num, translate_caa_type, types_from_id3

class CaaTypeTranslationTest(PicardTestCase):

    def test_translating_unknown_types_returns_input(self):
        if False:
            print('Hello World!')
        testtype = 'ThisIsAMadeUpCoverArtTypeName'
        self.assertEqual(translate_caa_type(testtype), testtype)

class Id3TypeTranslationTest(PicardTestCase):

    def test_image_type_from_id3_num(self):
        if False:
            return 10
        self.assertEqual(image_type_from_id3_num(0), 'other')
        self.assertEqual(image_type_from_id3_num(3), 'front')
        self.assertEqual(image_type_from_id3_num(6), 'medium')
        self.assertEqual(image_type_from_id3_num(9999), 'other')

    def test_image_type_as_id3_num(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(image_type_as_id3_num('other'), 0)
        self.assertEqual(image_type_as_id3_num('front'), 3)
        self.assertEqual(image_type_as_id3_num('medium'), 6)
        self.assertEqual(image_type_as_id3_num('track'), 6)
        self.assertEqual(image_type_as_id3_num('unknowntype'), 0)

    def test_types_from_id3(self):
        if False:
            print('Hello World!')
        self.assertEqual(types_from_id3(0), ['other'])
        self.assertEqual(types_from_id3(3), ['front'])
        self.assertEqual(types_from_id3(6), ['medium'])
        self.assertEqual(types_from_id3(9999), ['other'])