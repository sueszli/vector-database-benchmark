from xml.dom import minidom
from django.conf import settings
from django.contrib.sites.models import Site
from django.test import TestCase, modify_settings, override_settings
from .models import City

@modify_settings(INSTALLED_APPS={'append': 'django.contrib.sites'})
@override_settings(ROOT_URLCONF='gis_tests.geoapp.urls')
class GeoFeedTest(TestCase):
    fixtures = ['initial']

    @classmethod
    def setUpTestData(cls):
        if False:
            return 10
        Site(id=settings.SITE_ID, domain='example.com', name='example.com').save()

    def assertChildNodes(self, elem, expected):
        if False:
            i = 10
            return i + 15
        'Taken from syndication/tests.py.'
        actual = {n.nodeName for n in elem.childNodes}
        expected = set(expected)
        self.assertEqual(actual, expected)

    def test_geofeed_rss(self):
        if False:
            print('Hello World!')
        'Tests geographic feeds using GeoRSS over RSSv2.'
        doc1 = minidom.parseString(self.client.get('/feeds/rss1/').content)
        doc2 = minidom.parseString(self.client.get('/feeds/rss2/').content)
        (feed1, feed2) = (doc1.firstChild, doc2.firstChild)
        self.assertChildNodes(feed2.getElementsByTagName('channel')[0], ['title', 'link', 'description', 'language', 'lastBuildDate', 'item', 'georss:box', 'atom:link'])
        for feed in [feed1, feed2]:
            self.assertEqual(feed.getAttribute('xmlns:georss'), 'http://www.georss.org/georss')
            chan = feed.getElementsByTagName('channel')[0]
            items = chan.getElementsByTagName('item')
            self.assertEqual(len(items), City.objects.count())
            for item in items:
                self.assertChildNodes(item, ['title', 'link', 'description', 'guid', 'georss:point'])

    def test_geofeed_atom(self):
        if False:
            while True:
                i = 10
        'Testing geographic feeds using GeoRSS over Atom.'
        doc1 = minidom.parseString(self.client.get('/feeds/atom1/').content)
        doc2 = minidom.parseString(self.client.get('/feeds/atom2/').content)
        (feed1, feed2) = (doc1.firstChild, doc2.firstChild)
        self.assertChildNodes(feed2, ['title', 'link', 'id', 'updated', 'entry', 'georss:box'])
        for feed in [feed1, feed2]:
            self.assertEqual(feed.getAttribute('xmlns:georss'), 'http://www.georss.org/georss')
            entries = feed.getElementsByTagName('entry')
            self.assertEqual(len(entries), City.objects.count())
            for entry in entries:
                self.assertChildNodes(entry, ['title', 'link', 'id', 'summary', 'georss:point'])

    def test_geofeed_w3c(self):
        if False:
            return 10
        'Testing geographic feeds using W3C Geo.'
        doc = minidom.parseString(self.client.get('/feeds/w3cgeo1/').content)
        feed = doc.firstChild
        self.assertEqual(feed.getAttribute('xmlns:geo'), 'http://www.w3.org/2003/01/geo/wgs84_pos#')
        chan = feed.getElementsByTagName('channel')[0]
        items = chan.getElementsByTagName('item')
        self.assertEqual(len(items), City.objects.count())
        for item in items:
            self.assertChildNodes(item, ['title', 'link', 'description', 'guid', 'geo:lat', 'geo:lon'])
        with self.assertRaises(ValueError):
            self.client.get('/feeds/w3cgeo2/')
        with self.assertRaises(ValueError):
            self.client.get('/feeds/w3cgeo3/')