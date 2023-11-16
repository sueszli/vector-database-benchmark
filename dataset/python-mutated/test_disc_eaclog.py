from typing import Iterator
from test.picardtestcase import PicardTestCase, get_test_data_path
from picard.disc.eaclog import filter_toc_entries, toc_from_file
from picard.disc.utils import NotSupportedTOCError, TocEntry
test_log = ('TEST LOG', ' Track |   Start  |  Length  | Start sector | End sector', '---------------------------------------------------------', '    1  |  0:00.00 |  5:32.14 |         0    |    24913', '    2  |  5:32.14 |  4:07.22 |     24914    |    43460', '    3  |  9:39.36 |  3:50.29 |     43461    |    60739', '', 'foo')
test_entries = [TocEntry(1, 0, 24913), TocEntry(2, 24914, 43460), TocEntry(3, 43461, 60739)]

class TestFilterTocEntries(PicardTestCase):

    def test_filter_toc_entries(self):
        if False:
            for i in range(10):
                print('nop')
        result = filter_toc_entries(iter(test_log))
        self.assertTrue(isinstance(result, Iterator))
        entries = list(result)
        self.assertEqual(test_entries, entries)

class TestTocFromFile(PicardTestCase):

    def _test_toc_from_file(self, logfile):
        if False:
            print('Hello World!')
        test_log = get_test_data_path(logfile)
        toc = toc_from_file(test_log)
        self.assertEqual((1, 8, 149323, 150, 25064, 43611, 60890, 83090, 100000, 115057, 135558), toc)

    def test_toc_from_file_eac_utf8(self):
        if False:
            print('Hello World!')
        self._test_toc_from_file('eac-utf8.log')

    def test_toc_from_file_eac_utf16le(self):
        if False:
            print('Hello World!')
        self._test_toc_from_file('eac-utf16le.log')

    def test_toc_from_file_xld(self):
        if False:
            i = 10
            return i + 15
        self._test_toc_from_file('xld.log')

    def test_toc_from_file_freac(self):
        if False:
            print('Hello World!')
        test_log = get_test_data_path('freac.log')
        toc = toc_from_file(test_log)
        self.assertEqual((1, 10, 280995, 150, 27732, 54992, 82825, 108837, 125742, 155160, 181292, 213715, 245750), toc)

    def test_toc_from_file_with_datatrack(self):
        if False:
            i = 10
            return i + 15
        test_log = get_test_data_path('eac-datatrack.log')
        toc = toc_from_file(test_log)
        self.assertEqual((1, 8, 178288, 150, 20575, 42320, 62106, 78432, 94973, 109750, 130111), toc)

    def test_toc_from_file_with_datatrack_freac(self):
        if False:
            while True:
                i = 10
        test_log = get_test_data_path('freac-datatrack.log')
        toc = toc_from_file(test_log)
        self.assertEqual((1, 13, 218150, 150, 15014, 33313, 49023, 65602, 81316, 102381, 116294, 133820, 151293, 168952, 190187, 203916), toc)

    def test_toc_from_empty_file(self):
        if False:
            for i in range(10):
                print('nop')
        test_log = get_test_data_path('eac-empty.log')
        with self.assertRaises(NotSupportedTOCError):
            toc_from_file(test_log)