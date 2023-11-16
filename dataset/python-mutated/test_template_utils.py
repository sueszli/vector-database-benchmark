import unittest
import datetime
from email.utils import formatdate
from tests import NyaaTestCase
from nyaa.template_utils import _jinja2_filter_rfc822, _jinja2_filter_rfc822_es, get_utc_timestamp, get_display_time, timesince, filter_truthy, category_name

class TestTemplateUtils(NyaaTestCase):

    def test_filter_rfc822(self):
        if False:
            while True:
                i = 10
        test_date = datetime.datetime(2017, 2, 15, 11, 15, 34, 100, datetime.timezone.utc)
        self.assertEqual(_jinja2_filter_rfc822(test_date), 'Wed, 15 Feb 2017 11:15:34 -0000')

    def test_filter_rfc822_es(self):
        if False:
            return 10
        test_date_str = '2017-02-15T11:15:34'
        expected = formatdate(float(datetime.datetime(2017, 2, 15, 11, 15, 34, 100).timestamp()))
        self.assertEqual(_jinja2_filter_rfc822_es(test_date_str), expected)

    def test_get_utc_timestamp(self):
        if False:
            i = 10
            return i + 15
        test_date_str = '2017-02-15T11:15:34'
        self.assertEqual(get_utc_timestamp(test_date_str), 1487157334)

    def test_get_display_time(self):
        if False:
            return 10
        test_date_str = '2017-02-15T11:15:34'
        self.assertEqual(get_display_time(test_date_str), '2017-02-15 11:15')

    def test_timesince(self):
        if False:
            print('Hello World!')
        now = datetime.datetime.utcnow()
        self.assertEqual(timesince(now), 'just now')
        self.assertEqual(timesince(now - datetime.timedelta(seconds=5)), '5 seconds ago')
        self.assertEqual(timesince(now - datetime.timedelta(minutes=1)), '1 minute ago')
        self.assertEqual(timesince(now - datetime.timedelta(minutes=38, seconds=43)), '38 minutes ago')
        self.assertEqual(timesince(now - datetime.timedelta(hours=2, minutes=38, seconds=51)), '2 hours ago')
        bigger = now - datetime.timedelta(days=3)
        self.assertEqual(timesince(bigger), bigger.strftime('%Y-%m-%d %H:%M UTC'))

    @unittest.skip('Not yet implemented')
    def test_static_cachebuster(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip('Not yet implemented')
    def test_modify_query(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_filter_truthy(self):
        if False:
            print('Hello World!')
        my_list = [True, False, 'hello!', '', 1, 0, -1, 1.0, 0.0, -1.0, ['test'], [], {'marco': 'polo'}, {}, None]
        expected_result = [True, 'hello!', 1, -1, 1.0, -1.0, ['test'], {'marco': 'polo'}]
        self.assertListEqual(filter_truthy(my_list), expected_result)

    def test_category_name(self):
        if False:
            return 10
        with self.app_context:
            self.assertEqual(category_name('1_0'), 'Anime')
            self.assertEqual(category_name('1_2'), 'Anime - English-translated')
            self.assertEqual(category_name('100_0'), '???')
            self.assertEqual(category_name('1_100'), '???')
            self.assertEqual(category_name('0_0'), '???')
if __name__ == '__main__':
    unittest.main()