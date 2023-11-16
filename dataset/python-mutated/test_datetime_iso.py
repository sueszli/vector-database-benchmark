import unittest
import json
import jc.parsers.datetime_iso

class MyTests(unittest.TestCase):

    def test_datetime_iso_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'datetime_iso' with no data\n        "
        self.assertEqual(jc.parsers.datetime_iso.parse('', quiet=True), {})

    def test_datetime_iso_z(self):
        if False:
            print('Hello World!')
        '\n        Test ISO datetime string with Z timezone\n        '
        data = '2007-04-05T14:30Z'
        expected = json.loads('{"year":2007,"month":"Apr","month_num":4,"day":5,"weekday":"Thu","weekday_num":4,"hour":2,"hour_24":14,"minute":30,"second":0,"microsecond":0,"period":"PM","utc_offset":"+0000","day_of_year":95,"week_of_year":14,"iso":"2007-04-05T14:30:00+00:00","timestamp":1175783400}')
        self.assertEqual(jc.parsers.datetime_iso.parse(data, quiet=True), expected)

    def test_datetime_iso_microseconds(self):
        if False:
            i = 10
            return i + 15
        '\n        Test ISO datetime string with Z timezone\n        '
        data = '2007-04-05T14:30.555Z'
        expected = json.loads('{"year":2007,"month":"Apr","month_num":4,"day":5,"weekday":"Thu","weekday_num":4,"hour":2,"hour_24":14,"minute":0,"second":30,"microsecond":555000,"period":"PM","utc_offset":"+0000","day_of_year":95,"week_of_year":14,"iso":"2007-04-05T14:00:30.555000+00:00","timestamp":1175781630}')
        self.assertEqual(jc.parsers.datetime_iso.parse(data, quiet=True), expected)

    def test_datetime_iso_plus_offset(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test ISO datetime string with + offset\n        '
        data = '2007-04-05T14:30+03:30'
        expected = json.loads('{"year":2007,"month":"Apr","month_num":4,"day":5,"weekday":"Thu","weekday_num":4,"hour":2,"hour_24":14,"minute":30,"second":0,"microsecond":0,"period":"PM","utc_offset":"+0330","day_of_year":95,"week_of_year":14,"iso":"2007-04-05T14:30:00+03:30","timestamp":1175770800}')
        self.assertEqual(jc.parsers.datetime_iso.parse(data, quiet=True), expected)

    def test_datetime_iso_negative_offset(self):
        if False:
            i = 10
            return i + 15
        '\n        Test ISO datetime string with - offset\n        '
        data = '2007-04-05T14:30-03:30'
        expected = json.loads('{"year":2007,"month":"Apr","month_num":4,"day":5,"weekday":"Thu","weekday_num":4,"hour":2,"hour_24":14,"minute":30,"second":0,"microsecond":0,"period":"PM","utc_offset":"-0330","day_of_year":95,"week_of_year":14,"iso":"2007-04-05T14:30:00-03:30","timestamp":1175796000}')
        self.assertEqual(jc.parsers.datetime_iso.parse(data, quiet=True), expected)

    def test_datetime_iso_nocolon_offset(self):
        if False:
            i = 10
            return i + 15
        '\n        Test ISO datetime string with no colon offset\n        '
        data = '2007-04-05T14:30+0300'
        expected = json.loads('{"year":2007,"month":"Apr","month_num":4,"day":5,"weekday":"Thu","weekday_num":4,"hour":2,"hour_24":14,"minute":30,"second":0,"microsecond":0,"period":"PM","utc_offset":"+0300","day_of_year":95,"week_of_year":14,"iso":"2007-04-05T14:30:00+03:00","timestamp":1175772600}')
        self.assertEqual(jc.parsers.datetime_iso.parse(data, quiet=True), expected)
if __name__ == '__main__':
    unittest.main()