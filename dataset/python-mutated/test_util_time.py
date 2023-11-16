from test.picardtestcase import PicardTestCase
from picard.util.time import get_timestamp, seconds_to_dhms

class UtilTimeTest(PicardTestCase):

    def test_seconds_to_dhms(self):
        if False:
            i = 10
            return i + 15
        self.assertTupleEqual(seconds_to_dhms(0), (0, 0, 0, 0))
        self.assertTupleEqual(seconds_to_dhms(1), (0, 0, 0, 1))
        self.assertTupleEqual(seconds_to_dhms(60), (0, 0, 1, 0))
        self.assertTupleEqual(seconds_to_dhms(61), (0, 0, 1, 1))
        self.assertTupleEqual(seconds_to_dhms(120), (0, 0, 2, 0))
        self.assertTupleEqual(seconds_to_dhms(3599), (0, 0, 59, 59))
        self.assertTupleEqual(seconds_to_dhms(3600), (0, 1, 0, 0))
        self.assertTupleEqual(seconds_to_dhms(3601), (0, 1, 0, 1))
        self.assertTupleEqual(seconds_to_dhms(3660), (0, 1, 1, 0))
        self.assertTupleEqual(seconds_to_dhms(3661), (0, 1, 1, 1))
        self.assertTupleEqual(seconds_to_dhms(86399), (0, 23, 59, 59))
        self.assertTupleEqual(seconds_to_dhms(86400), (1, 0, 0, 0))

    def test_get_timestamp(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(get_timestamp(0), '')
        self.assertEqual(get_timestamp(1), '01s')
        self.assertEqual(get_timestamp(60), '01m 00s')
        self.assertEqual(get_timestamp(61), '01m 01s')
        self.assertEqual(get_timestamp(120), '02m 00s')
        self.assertEqual(get_timestamp(3599), '59m 59s')
        self.assertEqual(get_timestamp(3600), '01h 00m')
        self.assertEqual(get_timestamp(3601), '01h 00m')
        self.assertEqual(get_timestamp(3660), '01h 01m')
        self.assertEqual(get_timestamp(3661), '01h 01m')
        self.assertEqual(get_timestamp(86399), '23h 59m')
        self.assertEqual(get_timestamp(86400), '01d 00h')