import datetime
from django.test import SimpleTestCase
from django.utils import feedgenerator
from django.utils.timezone import get_fixed_timezone

class FeedgeneratorTests(SimpleTestCase):
    """
    Tests for the low-level syndication feed framework.
    """

    def test_get_tag_uri(self):
        if False:
            while True:
                i = 10
        '\n        get_tag_uri() correctly generates TagURIs.\n        '
        self.assertEqual(feedgenerator.get_tag_uri('http://example.org/foo/bar#headline', datetime.date(2004, 10, 25)), 'tag:example.org,2004-10-25:/foo/bar/headline')

    def test_get_tag_uri_with_port(self):
        if False:
            return 10
        '\n        get_tag_uri() correctly generates TagURIs from URLs with port numbers.\n        '
        self.assertEqual(feedgenerator.get_tag_uri('http://www.example.org:8000/2008/11/14/django#headline', datetime.datetime(2008, 11, 14, 13, 37, 0)), 'tag:www.example.org,2008-11-14:/2008/11/14/django/headline')

    def test_rfc2822_date(self):
        if False:
            return 10
        '\n        rfc2822_date() correctly formats datetime objects.\n        '
        self.assertEqual(feedgenerator.rfc2822_date(datetime.datetime(2008, 11, 14, 13, 37, 0)), 'Fri, 14 Nov 2008 13:37:00 -0000')

    def test_rfc2822_date_with_timezone(self):
        if False:
            i = 10
            return i + 15
        '\n        rfc2822_date() correctly formats datetime objects with tzinfo.\n        '
        self.assertEqual(feedgenerator.rfc2822_date(datetime.datetime(2008, 11, 14, 13, 37, 0, tzinfo=get_fixed_timezone(60))), 'Fri, 14 Nov 2008 13:37:00 +0100')

    def test_rfc2822_date_without_time(self):
        if False:
            return 10
        '\n        rfc2822_date() correctly formats date objects.\n        '
        self.assertEqual(feedgenerator.rfc2822_date(datetime.date(2008, 11, 14)), 'Fri, 14 Nov 2008 00:00:00 -0000')

    def test_rfc3339_date(self):
        if False:
            return 10
        '\n        rfc3339_date() correctly formats datetime objects.\n        '
        self.assertEqual(feedgenerator.rfc3339_date(datetime.datetime(2008, 11, 14, 13, 37, 0)), '2008-11-14T13:37:00Z')

    def test_rfc3339_date_with_timezone(self):
        if False:
            i = 10
            return i + 15
        '\n        rfc3339_date() correctly formats datetime objects with tzinfo.\n        '
        self.assertEqual(feedgenerator.rfc3339_date(datetime.datetime(2008, 11, 14, 13, 37, 0, tzinfo=get_fixed_timezone(120))), '2008-11-14T13:37:00+02:00')

    def test_rfc3339_date_without_time(self):
        if False:
            i = 10
            return i + 15
        '\n        rfc3339_date() correctly formats date objects.\n        '
        self.assertEqual(feedgenerator.rfc3339_date(datetime.date(2008, 11, 14)), '2008-11-14T00:00:00Z')

    def test_atom1_mime_type(self):
        if False:
            return 10
        '\n        Atom MIME type has UTF8 Charset parameter set\n        '
        atom_feed = feedgenerator.Atom1Feed('title', 'link', 'description')
        self.assertEqual(atom_feed.content_type, 'application/atom+xml; charset=utf-8')

    def test_rss_mime_type(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        RSS MIME type has UTF8 Charset parameter set\n        '
        rss_feed = feedgenerator.Rss201rev2Feed('title', 'link', 'description')
        self.assertEqual(rss_feed.content_type, 'application/rss+xml; charset=utf-8')

    def test_feed_without_feed_url_gets_rendered_without_atom_link(self):
        if False:
            return 10
        feed = feedgenerator.Rss201rev2Feed('title', '/link/', 'descr')
        self.assertIsNone(feed.feed['feed_url'])
        feed_content = feed.writeString('utf-8')
        self.assertNotIn('<atom:link', feed_content)
        self.assertNotIn('href="/feed/"', feed_content)
        self.assertNotIn('rel="self"', feed_content)

    def test_feed_with_feed_url_gets_rendered_with_atom_link(self):
        if False:
            while True:
                i = 10
        feed = feedgenerator.Rss201rev2Feed('title', '/link/', 'descr', feed_url='/feed/')
        self.assertEqual(feed.feed['feed_url'], '/feed/')
        feed_content = feed.writeString('utf-8')
        self.assertIn('<atom:link', feed_content)
        self.assertIn('href="/feed/"', feed_content)
        self.assertIn('rel="self"', feed_content)

    def test_atom_add_item(self):
        if False:
            return 10
        feed = feedgenerator.Atom1Feed('title', '/link/', 'descr')
        feed.add_item('item_title', 'item_link', 'item_description')
        feed.writeString('utf-8')

    def test_deterministic_attribute_order(self):
        if False:
            i = 10
            return i + 15
        feed = feedgenerator.Atom1Feed('title', '/link/', 'desc')
        feed_content = feed.writeString('utf-8')
        self.assertIn('href="/link/" rel="alternate"', feed_content)

    def test_latest_post_date_returns_utc_time(self):
        if False:
            i = 10
            return i + 15
        for use_tz in (True, False):
            with self.settings(USE_TZ=use_tz):
                rss_feed = feedgenerator.Rss201rev2Feed('title', 'link', 'description')
                self.assertEqual(rss_feed.latest_post_date().tzinfo, datetime.timezone.utc)