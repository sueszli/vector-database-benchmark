from test.picardtestcase import PicardTestCase, get_test_data_path
from picard.disc.whipperlog import toc_from_file

class TestTocFromFile(PicardTestCase):

    def test_toc_from_file(self):
        if False:
            i = 10
            return i + 15
        test_log = get_test_data_path('whipper.log')
        toc = toc_from_file(test_log)
        self.assertEqual((1, 8, 149323, 150, 25064, 43611, 60890, 83090, 100000, 115057, 135558), toc)