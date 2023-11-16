import unittest
from scrapy.utils.sitemap import Sitemap, sitemap_urls_from_robots

class SitemapTest(unittest.TestCase):

    def test_sitemap(self):
        if False:
            return 10
        s = Sitemap(b'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.google.com/schemas/sitemap/0.84">\n  <url>\n    <loc>http://www.example.com/</loc>\n    <lastmod>2009-08-16</lastmod>\n    <changefreq>daily</changefreq>\n    <priority>1</priority>\n  </url>\n  <url>\n    <loc>http://www.example.com/Special-Offers.html</loc>\n    <lastmod>2009-08-16</lastmod>\n    <changefreq>weekly</changefreq>\n    <priority>0.8</priority>\n  </url>\n</urlset>')
        assert s.type == 'urlset'
        self.assertEqual(list(s), [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'}, {'priority': '0.8', 'loc': 'http://www.example.com/Special-Offers.html', 'lastmod': '2009-08-16', 'changefreq': 'weekly'}])

    def test_sitemap_index(self):
        if False:
            return 10
        s = Sitemap(b'<?xml version="1.0" encoding="UTF-8"?>\n<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n   <sitemap>\n      <loc>http://www.example.com/sitemap1.xml.gz</loc>\n      <lastmod>2004-10-01T18:23:17+00:00</lastmod>\n   </sitemap>\n   <sitemap>\n      <loc>http://www.example.com/sitemap2.xml.gz</loc>\n      <lastmod>2005-01-01</lastmod>\n   </sitemap>\n</sitemapindex>')
        assert s.type == 'sitemapindex'
        self.assertEqual(list(s), [{'loc': 'http://www.example.com/sitemap1.xml.gz', 'lastmod': '2004-10-01T18:23:17+00:00'}, {'loc': 'http://www.example.com/sitemap2.xml.gz', 'lastmod': '2005-01-01'}])

    def test_sitemap_strip(self):
        if False:
            print('Hello World!')
        "Assert we can deal with trailing spaces inside <loc> tags - we've\n        seen those\n        "
        s = Sitemap(b'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.google.com/schemas/sitemap/0.84">\n  <url>\n    <loc> http://www.example.com/</loc>\n    <lastmod>2009-08-16</lastmod>\n    <changefreq>daily</changefreq>\n    <priority>1</priority>\n  </url>\n  <url>\n    <loc> http://www.example.com/2</loc>\n    <lastmod />\n  </url>\n</urlset>\n')
        self.assertEqual(list(s), [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'}, {'loc': 'http://www.example.com/2', 'lastmod': ''}])

    def test_sitemap_wrong_ns(self):
        if False:
            print('Hello World!')
        'We have seen sitemaps with wrongs ns. Presumably, Google still works\n        with these, though is not 100% confirmed'
        s = Sitemap(b'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.google.com/schemas/sitemap/0.84">\n  <url xmlns="">\n    <loc> http://www.example.com/</loc>\n    <lastmod>2009-08-16</lastmod>\n    <changefreq>daily</changefreq>\n    <priority>1</priority>\n  </url>\n  <url xmlns="">\n    <loc> http://www.example.com/2</loc>\n    <lastmod />\n  </url>\n</urlset>\n')
        self.assertEqual(list(s), [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'}, {'loc': 'http://www.example.com/2', 'lastmod': ''}])

    def test_sitemap_wrong_ns2(self):
        if False:
            return 10
        'We have seen sitemaps with wrongs ns. Presumably, Google still works\n        with these, though is not 100% confirmed'
        s = Sitemap(b'<?xml version="1.0" encoding="UTF-8"?>\n<urlset>\n  <url xmlns="">\n    <loc> http://www.example.com/</loc>\n    <lastmod>2009-08-16</lastmod>\n    <changefreq>daily</changefreq>\n    <priority>1</priority>\n  </url>\n  <url xmlns="">\n    <loc> http://www.example.com/2</loc>\n    <lastmod />\n  </url>\n</urlset>\n')
        assert s.type == 'urlset'
        self.assertEqual(list(s), [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'}, {'loc': 'http://www.example.com/2', 'lastmod': ''}])

    def test_sitemap_urls_from_robots(self):
        if False:
            for i in range(10):
                print('nop')
        robots = 'User-agent: *\nDisallow: /aff/\nDisallow: /wl/\n\n# Search and shopping refining\nDisallow: /s*/*facet\nDisallow: /s*/*tags\n\n# Sitemap files\nSitemap: http://example.com/sitemap.xml\nSitemap: http://example.com/sitemap-product-index.xml\nSitemap: HTTP://example.com/sitemap-uppercase.xml\nSitemap: /sitemap-relative-url.xml\n\n# Forums\nDisallow: /forum/search/\nDisallow: /forum/active/\n'
        self.assertEqual(list(sitemap_urls_from_robots(robots, base_url='http://example.com')), ['http://example.com/sitemap.xml', 'http://example.com/sitemap-product-index.xml', 'http://example.com/sitemap-uppercase.xml', 'http://example.com/sitemap-relative-url.xml'])

    def test_sitemap_blanklines(self):
        if False:
            return 10
        'Assert we can deal with starting blank lines before <xml> tag'
        s = Sitemap(b'\n<?xml version="1.0" encoding="UTF-8"?>\n<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n\n<!-- cache: cached = yes name = sitemap_jspCache key = sitemap -->\n<sitemap>\n<loc>http://www.example.com/sitemap1.xml</loc>\n<lastmod>2013-07-15</lastmod>\n</sitemap>\n\n<sitemap>\n<loc>http://www.example.com/sitemap2.xml</loc>\n<lastmod>2013-07-15</lastmod>\n</sitemap>\n\n<sitemap>\n<loc>http://www.example.com/sitemap3.xml</loc>\n<lastmod>2013-07-15</lastmod>\n</sitemap>\n\n<!-- end cache -->\n</sitemapindex>\n')
        self.assertEqual(list(s), [{'lastmod': '2013-07-15', 'loc': 'http://www.example.com/sitemap1.xml'}, {'lastmod': '2013-07-15', 'loc': 'http://www.example.com/sitemap2.xml'}, {'lastmod': '2013-07-15', 'loc': 'http://www.example.com/sitemap3.xml'}])

    def test_comment(self):
        if False:
            for i in range(10):
                print('nop')
        s = Sitemap(b'<?xml version="1.0" encoding="UTF-8"?>\n    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"\n        xmlns:xhtml="http://www.w3.org/1999/xhtml">\n        <url>\n            <loc>http://www.example.com/</loc>\n            <!-- this is a comment on which the parser might raise an exception if implemented incorrectly -->\n        </url>\n    </urlset>')
        self.assertEqual(list(s), [{'loc': 'http://www.example.com/'}])

    def test_alternate(self):
        if False:
            while True:
                i = 10
        s = Sitemap(b'<?xml version="1.0" encoding="UTF-8"?>\n    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"\n        xmlns:xhtml="http://www.w3.org/1999/xhtml">\n        <url>\n            <loc>http://www.example.com/english/</loc>\n            <xhtml:link rel="alternate" hreflang="de"\n                href="http://www.example.com/deutsch/"/>\n            <xhtml:link rel="alternate" hreflang="de-ch"\n                href="http://www.example.com/schweiz-deutsch/"/>\n            <xhtml:link rel="alternate" hreflang="en"\n                href="http://www.example.com/english/"/>\n            <xhtml:link rel="alternate" hreflang="en"/><!-- wrong tag without href -->\n        </url>\n    </urlset>')
        self.assertEqual(list(s), [{'loc': 'http://www.example.com/english/', 'alternate': ['http://www.example.com/deutsch/', 'http://www.example.com/schweiz-deutsch/', 'http://www.example.com/english/']}])

    def test_xml_entity_expansion(self):
        if False:
            while True:
                i = 10
        s = Sitemap(b'<?xml version="1.0" encoding="utf-8"?>\n          <!DOCTYPE foo [\n          <!ELEMENT foo ANY >\n          <!ENTITY xxe SYSTEM "file:///etc/passwd" >\n          ]>\n          <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n            <url>\n              <loc>http://127.0.0.1:8000/&xxe;</loc>\n            </url>\n          </urlset>\n        ')
        self.assertEqual(list(s), [{'loc': 'http://127.0.0.1:8000/'}])
if __name__ == '__main__':
    unittest.main()