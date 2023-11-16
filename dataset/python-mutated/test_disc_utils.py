from test.picardtestcase import PicardTestCase
from picard.disc.utils import NotSupportedTOCError, TocEntry, calculate_mb_toc_numbers
test_entries = [TocEntry(1, 0, 24913), TocEntry(2, 24914, 43460), TocEntry(3, 43461, 60739)]

class TestCalculateMbTocNumbers(PicardTestCase):

    def test_calculate_mb_toc_numbers(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual((1, 3, 60890, 150, 25064, 43611), calculate_mb_toc_numbers(test_entries))

    def test_calculate_mb_toc_numbers_invalid_track_numbers(self):
        if False:
            return 10
        entries = [TocEntry(1, 0, 100), TocEntry(3, 101, 200), TocEntry(4, 201, 300)]
        with self.assertRaisesRegex(NotSupportedTOCError, '^Non-standard track number sequence: \\(1, 3, 4\\)$'):
            calculate_mb_toc_numbers(entries)

    def test_calculate_mb_toc_numbers_empty_entries(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(NotSupportedTOCError, '^Empty track list$'):
            calculate_mb_toc_numbers([])

    def test_calculate_mb_toc_numbers_ignore_datatrack(self):
        if False:
            i = 10
            return i + 15
        entries = [*test_entries, TocEntry(4, 72140, 80000)]
        self.assertEqual((1, 3, 60890, 150, 25064, 43611), calculate_mb_toc_numbers(entries))