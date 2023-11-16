from test.picardtestcase import PicardTestCase
from picard.const.sys import IS_WIN
from picard.tagger import ParseItemsToLoad

class TestMessageParsing(PicardTestCase):

    def test(self):
        if False:
            i = 10
            return i + 15
        test_cases = {'test_case.mp3', 'file:///home/picard/music/test.flac', 'mbid://recording/7cd3782d-86dc-4dd1-8d9b-e37f9cbe6b94', 'https://musicbrainz.org/recording/7cd3782d-86dc-4dd1-8d9b-e37f9cbe6b94', 'http://musicbrainz.org/recording/7cd3782d-86dc-4dd1-8d9b-e37f9cbe6b94'}
        result = ParseItemsToLoad(test_cases)
        self.assertSetEqual(result.files, {'test_case.mp3', '/home/picard/music/test.flac'}, 'Files test')
        self.assertSetEqual(result.mbids, {'recording/7cd3782d-86dc-4dd1-8d9b-e37f9cbe6b94'}, 'MBIDs test')
        self.assertSetEqual(result.urls, {'recording/7cd3782d-86dc-4dd1-8d9b-e37f9cbe6b94', 'recording/7cd3782d-86dc-4dd1-8d9b-e37f9cbe6b94'}, 'URLs test')

    def test_bool_files_true(self):
        if False:
            print('Hello World!')
        test_cases = {'test_case.mp3'}
        self.assertTrue(ParseItemsToLoad(test_cases))

    def test_bool_mbids_true(self):
        if False:
            i = 10
            return i + 15
        test_cases = {'mbid://recording/7cd3782d-86dc-4dd1-8d9b-e37f9cbe6b94'}
        self.assertTrue(ParseItemsToLoad(test_cases))

    def test_bool_urls_true(self):
        if False:
            print('Hello World!')
        test_cases = {'https://musicbrainz.org/recording/7cd3782d-86dc-4dd1-8d9b-e37f9cbe6b94'}
        self.assertTrue(ParseItemsToLoad(test_cases))

    def test_bool_invalid_false(self):
        if False:
            while True:
                i = 10
        test_cases = {'mbd://recording/7cd3782d-86dc-4dd1-8d9b-e37f9cbe6b94'}
        self.assertFalse(ParseItemsToLoad(test_cases))

    def test_bool_empty_false(self):
        if False:
            print('Hello World!')
        test_cases = {}
        self.assertFalse(ParseItemsToLoad(test_cases))

    def test_windows_file_with_drive(self):
        if False:
            i = 10
            return i + 15
        test_cases = {'C:\\test_case.mp3'}
        if IS_WIN:
            self.assertTrue(ParseItemsToLoad(test_cases))
        else:
            self.assertFalse(ParseItemsToLoad(test_cases))