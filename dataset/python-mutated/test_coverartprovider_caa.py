from test.picardtestcase import PicardTestCase
from picard.coverart.providers.caa import caa_url_fallback_list

class CoverArtImageProviderCaaTest(PicardTestCase):

    def test_caa_url_fallback_list(self):
        if False:
            i = 10
            return i + 15

        def do_tests(sizes, expectations):
            if False:
                while True:
                    i = 10
            thumbnails = {size: 'url %s' % size for size in sizes}
            msgfmt = 'for size %s, with sizes %r, got %r, expected %r'
            for (size, expect) in expectations.items():
                result = caa_url_fallback_list(size, thumbnails)
                self.assertEqual(result, expect, msg=msgfmt % (size, sizes, result, expect))
        sizes = ('250', '500', '1200', 'large', 'small')
        expectations = {50: [], 250: ['url 250'], 400: ['url 250'], 500: ['url 500', 'url 250'], 600: ['url 500', 'url 250'], 1200: ['url 1200', 'url 500', 'url 250'], 1500: ['url 1200', 'url 500', 'url 250']}
        do_tests(sizes, expectations)
        sizes = ('250', '500', 'large', 'small')
        expectations = {50: [], 250: ['url 250'], 400: ['url 250'], 500: ['url 500', 'url 250'], 600: ['url 500', 'url 250'], 1200: ['url 500', 'url 250'], 1500: ['url 500', 'url 250']}
        do_tests(sizes, expectations)
        sizes = ('small', 'large', '1200', '2000', 'unknownsize')
        expectations = {50: [], 250: ['url small'], 400: ['url small'], 500: ['url large', 'url small'], 600: ['url large', 'url small'], 1200: ['url 1200', 'url large', 'url small'], 1500: ['url 1200', 'url large', 'url small']}
        do_tests(sizes, expectations)
        with self.assertRaises(TypeError):
            caa_url_fallback_list('not_an_integer', {'250': 'url 250'})
        with self.assertRaises(AttributeError):
            caa_url_fallback_list(250, 666)