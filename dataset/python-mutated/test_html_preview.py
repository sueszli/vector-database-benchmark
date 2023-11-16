from synapse.media.preview_html import _get_html_media_encodings, decode_body, parse_html_to_open_graph, summarize_paragraphs
from tests import unittest
try:
    import lxml
except ImportError:
    lxml = None

class SummarizeTestCase(unittest.TestCase):
    if not lxml:
        skip = 'url preview feature requires lxml'

    def test_long_summarize(self) -> None:
        if False:
            i = 10
            return i + 15
        example_paras = ['Tromsø (Norwegian pronunciation: [ˈtrʊmsœ] ( listen); Northern Sami:\n            Romsa; Finnish: Tromssa[2] Kven: Tromssa) is a city and municipality in\n            Troms county, Norway. The administrative centre of the municipality is\n            the city of Tromsø. Outside of Norway, Tromso and Tromsö are\n            alternative spellings of the city.Tromsø is considered the northernmost\n            city in the world with a population above 50,000. The most populous town\n            north of it is Alta, Norway, with a population of 14,272 (2013).', 'Tromsø lies in Northern Norway. The municipality has a population of\n            (2015) 72,066, but with an annual influx of students it has over 75,000\n            most of the year. It is the largest urban area in Northern Norway and the\n            third largest north of the Arctic Circle (following Murmansk and Norilsk).\n            Most of Tromsø, including the city centre, is located on the island of\n            Tromsøya, 350 kilometres (217 mi) north of the Arctic Circle. In 2012,\n            Tromsøya had a population of 36,088. Substantial parts of the urban area\n            are also situated on the mainland to the east, and on parts of Kvaløya—a\n            large island to the west. Tromsøya is connected to the mainland by the Tromsø\n            Bridge and the Tromsøysund Tunnel, and to the island of Kvaløya by the\n            Sandnessund Bridge. Tromsø Airport connects the city to many destinations\n            in Europe. The city is warmer than most other places located on the same\n            latitude, due to the warming effect of the Gulf Stream.', "The city centre of Tromsø contains the highest number of old wooden\n            houses in Northern Norway, the oldest house dating from 1789. The Arctic\n            Cathedral, a modern church from 1965, is probably the most famous landmark\n            in Tromsø. The city is a cultural centre for its region, with several\n            festivals taking place in the summer. Some of Norway's best-known\n             musicians, Torbjørn Brundtland and Svein Berge of the electronica duo\n             Röyksopp and Lene Marlin grew up and started their careers in Tromsø.\n             Noted electronic musician Geir Jenssen also hails from Tromsø."]
        desc = summarize_paragraphs(example_paras, min_size=200, max_size=500)
        self.assertEqual(desc, 'Tromsø (Norwegian pronunciation: [ˈtrʊmsœ] ( listen); Northern Sami: Romsa; Finnish: Tromssa[2] Kven: Tromssa) is a city and municipality in Troms county, Norway. The administrative centre of the municipality is the city of Tromsø. Outside of Norway, Tromso and Tromsö are alternative spellings of the city.Tromsø is considered the northernmost city in the world with a population above 50,000. The most populous town north of it is Alta, Norway, with a population of 14,272 (2013).')
        desc = summarize_paragraphs(example_paras[1:], min_size=200, max_size=500)
        self.assertEqual(desc, 'Tromsø lies in Northern Norway. The municipality has a population of (2015) 72,066, but with an annual influx of students it has over 75,000 most of the year. It is the largest urban area in Northern Norway and the third largest north of the Arctic Circle (following Murmansk and Norilsk). Most of Tromsø, including the city centre, is located on the island of Tromsøya, 350 kilometres (217 mi) north of the Arctic Circle. In 2012, Tromsøya had a population of 36,088. Substantial parts of the urban…')

    def test_short_summarize(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        example_paras = ['Tromsø (Norwegian pronunciation: [ˈtrʊmsœ] ( listen); Northern Sami: Romsa; Finnish: Tromssa[2] Kven: Tromssa) is a city and municipality in Troms county, Norway.', 'Tromsø lies in Northern Norway. The municipality has a population of (2015) 72,066, but with an annual influx of students it has over 75,000 most of the year.', 'The city centre of Tromsø contains the highest number of old wooden houses in Northern Norway, the oldest house dating from 1789. The Arctic Cathedral, a modern church from 1965, is probably the most famous landmark in Tromsø.']
        desc = summarize_paragraphs(example_paras, min_size=200, max_size=500)
        self.assertEqual(desc, 'Tromsø (Norwegian pronunciation: [ˈtrʊmsœ] ( listen); Northern Sami: Romsa; Finnish: Tromssa[2] Kven: Tromssa) is a city and municipality in Troms county, Norway.\n\nTromsø lies in Northern Norway. The municipality has a population of (2015) 72,066, but with an annual influx of students it has over 75,000 most of the year.')

    def test_small_then_large_summarize(self) -> None:
        if False:
            i = 10
            return i + 15
        example_paras = ['Tromsø (Norwegian pronunciation: [ˈtrʊmsœ] ( listen); Northern Sami: Romsa; Finnish: Tromssa[2] Kven: Tromssa) is a city and municipality in Troms county, Norway.', 'Tromsø lies in Northern Norway. The municipality has a population of (2015) 72,066, but with an annual influx of students it has over 75,000 most of the year. The city centre of Tromsø contains the highest number of old wooden houses in Northern Norway, the oldest house dating from 1789. The Arctic Cathedral, a modern church from 1965, is probably the most famous landmark in Tromsø.']
        desc = summarize_paragraphs(example_paras, min_size=200, max_size=500)
        self.assertEqual(desc, 'Tromsø (Norwegian pronunciation: [ˈtrʊmsœ] ( listen); Northern Sami: Romsa; Finnish: Tromssa[2] Kven: Tromssa) is a city and municipality in Troms county, Norway.\n\nTromsø lies in Northern Norway. The municipality has a population of (2015) 72,066, but with an annual influx of students it has over 75,000 most of the year. The city centre of Tromsø contains the highest number of old wooden houses in Northern Norway, the oldest house dating from 1789. The Arctic Cathedral, a modern church from…')

class OpenGraphFromHtmlTestCase(unittest.TestCase):
    if not lxml:
        skip = 'url preview feature requires lxml'

    def test_simple(self) -> None:
        if False:
            return 10
        html = b'\n        <html>\n        <head><title>Foo</title></head>\n        <body>\n        Some text.\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': 'Foo', 'og:description': 'Some text.'})

    def test_comment(self) -> None:
        if False:
            i = 10
            return i + 15
        html = b'\n        <html>\n        <head><title>Foo</title></head>\n        <body>\n        <!-- HTML comment -->\n        Some text.\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': 'Foo', 'og:description': 'Some text.'})

    def test_comment2(self) -> None:
        if False:
            print('Hello World!')
        html = b'\n        <html>\n        <head><title>Foo</title></head>\n        <body>\n        Some text.\n        <!-- HTML comment -->\n        Some more text.\n        <p>Text</p>\n        More text\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': 'Foo', 'og:description': 'Some text.\n\nSome more text.\n\nText\n\nMore text'})

    def test_script(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        html = b'\n        <html>\n        <head><title>Foo</title></head>\n        <body>\n        <script> (function() {})() </script>\n        Some text.\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': 'Foo', 'og:description': 'Some text.'})

    def test_missing_title(self) -> None:
        if False:
            return 10
        html = b'\n        <html>\n        <body>\n        Some text.\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': None, 'og:description': 'Some text.'})
        html = b'\n        <html>\n        <head><title></title></head>\n        <body>\n        <h1>Title</h1>\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': 'Title', 'og:description': 'Title'})

    def test_h1_as_title(self) -> None:
        if False:
            return 10
        html = b'\n        <html>\n        <meta property="og:description" content="Some text."/>\n        <body>\n        <h1>Title</h1>\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': 'Title', 'og:description': 'Some text.'})

    def test_empty_description(self) -> None:
        if False:
            return 10
        'Description tags with empty content should be ignored.'
        html = b'\n        <html>\n        <meta property="og:description" content=""/>\n        <meta property="og:description"/>\n        <meta name="description" content=""/>\n        <meta name="description"/>\n        <meta name="description" content="Finally!"/>\n        <body>\n        <h1>Title</h1>\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': 'Title', 'og:description': 'Finally!'})

    def test_missing_title_and_broken_h1(self) -> None:
        if False:
            i = 10
            return i + 15
        html = b'\n        <html>\n        <body>\n        <h1><a href="foo"/></h1>\n        Some text.\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': None, 'og:description': 'Some text.'})

    def test_empty(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test a body with no data in it.'
        html = b''
        tree = decode_body(html, 'http://example.com/test.html')
        self.assertIsNone(tree)

    def test_no_tree(self) -> None:
        if False:
            while True:
                i = 10
        'A valid body with no tree in it.'
        html = b'\x00'
        tree = decode_body(html, 'http://example.com/test.html')
        self.assertIsNone(tree)

    def test_xml(self) -> None:
        if False:
            return 10
        'Test decoding XML and ensure it works properly.'
        html = b'\n        <?xml version="1.0" encoding="UTF-8"?>\n\n        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n        <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">\n        <head><title>Foo</title></head><body>Some text.</body></html>\n        '.strip()
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': 'Foo', 'og:description': 'Some text.'})

    def test_invalid_encoding(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'An invalid character encoding should be ignored and treated as UTF-8, if possible.'
        html = b'\n        <html>\n        <head><title>Foo</title></head>\n        <body>\n        Some text.\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html', 'invalid-encoding')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': 'Foo', 'og:description': 'Some text.'})

    def test_invalid_encoding2(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "A body which doesn't match the sent character encoding."
        html = b'\n        <html>\n        <head><title>\xff\xff Foo</title></head>\n        <body>\n        Some text.\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': 'ÿÿ Foo', 'og:description': 'Some text.'})

    def test_windows_1252(self) -> None:
        if False:
            i = 10
            return i + 15
        "A body which uses cp1252, but doesn't declare that."
        html = b'\n        <html>\n        <head><title>\xf3</title></head>\n        <body>\n        Some text.\n        </body>\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': 'ó', 'og:description': 'Some text.'})

    def test_twitter_tag(self) -> None:
        if False:
            i = 10
            return i + 15
        'Twitter card tags should be used if nothing else is available.'
        html = b'\n        <html>\n        <meta name="twitter:card" content="summary">\n        <meta name="twitter:description" content="Description">\n        <meta name="twitter:site" content="@matrixdotorg">\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': None, 'og:description': 'Description', 'og:site_name': '@matrixdotorg'})
        html = b'\n        <html>\n        <meta name="twitter:card" content="summary">\n        <meta name="twitter:description" content="Description">\n        <meta property="og:description" content="Real Description">\n        <meta name="twitter:site" content="@matrixdotorg">\n        <meta property="og:site_name" content="matrix.org">\n        </html>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': None, 'og:description': 'Real Description', 'og:site_name': 'matrix.org'})

    def test_nested_nodes(self) -> None:
        if False:
            while True:
                i = 10
        "A body with some nested nodes. Tests that we iterate over children\n        in the right order (and don't reverse the order of the text)."
        html = b'\n        <a href="somewhere">Welcome <b>the bold <u>and underlined text <svg>\n        with a cheeky SVG</svg></u> and <strong>some</strong> tail text</b></a>\n        '
        tree = decode_body(html, 'http://example.com/test.html')
        assert tree is not None
        og = parse_html_to_open_graph(tree)
        self.assertEqual(og, {'og:title': None, 'og:description': 'Welcome\n\nthe bold\n\nand underlined text\n\nand\n\nsome\n\ntail text'})

class MediaEncodingTestCase(unittest.TestCase):

    def test_meta_charset(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'A character encoding is found via the meta tag.'
        encodings = _get_html_media_encodings(b'\n        <html>\n        <head><meta charset="ascii">\n        </head>\n        </html>\n        ', 'text/html')
        self.assertEqual(list(encodings), ['ascii', 'utf-8', 'cp1252'])
        encodings = _get_html_media_encodings(b'\n        <html>\n        <head>< meta charset = ascii>\n        </head>\n        </html>\n        ', 'text/html')
        self.assertEqual(list(encodings), ['ascii', 'utf-8', 'cp1252'])

    def test_meta_charset_underscores(self) -> None:
        if False:
            while True:
                i = 10
        'A character encoding contains underscore.'
        encodings = _get_html_media_encodings(b'\n        <html>\n        <head><meta charset="Shift_JIS">\n        </head>\n        </html>\n        ', 'text/html')
        self.assertEqual(list(encodings), ['shift_jis', 'utf-8', 'cp1252'])

    def test_xml_encoding(self) -> None:
        if False:
            while True:
                i = 10
        'A character encoding is found via the meta tag.'
        encodings = _get_html_media_encodings(b'\n        <?xml version="1.0" encoding="ascii"?>\n        <html>\n        </html>\n        ', 'text/html')
        self.assertEqual(list(encodings), ['ascii', 'utf-8', 'cp1252'])

    def test_meta_xml_encoding(self) -> None:
        if False:
            while True:
                i = 10
        'Meta tags take precedence over XML encoding.'
        encodings = _get_html_media_encodings(b'\n        <?xml version="1.0" encoding="ascii"?>\n        <html>\n        <head><meta charset="UTF-16">\n        </head>\n        </html>\n        ', 'text/html')
        self.assertEqual(list(encodings), ['utf-16', 'ascii', 'utf-8', 'cp1252'])

    def test_content_type(self) -> None:
        if False:
            print('Hello World!')
        'A character encoding is found via the Content-Type header.'
        headers = ('text/html; charset="ascii";', 'text/html;charset=ascii;', 'text/html;  charset="ascii"', 'text/html; charset=ascii', 'text/html; charset="ascii;', 'text/html; charset=ascii";')
        for header in headers:
            encodings = _get_html_media_encodings(b'', header)
            self.assertEqual(list(encodings), ['ascii', 'utf-8', 'cp1252'])

    def test_fallback(self) -> None:
        if False:
            return 10
        'A character encoding cannot be found in the body or header.'
        encodings = _get_html_media_encodings(b'', 'text/html')
        self.assertEqual(list(encodings), ['utf-8', 'cp1252'])

    def test_duplicates(self) -> None:
        if False:
            while True:
                i = 10
        'Ensure each encoding is only attempted once.'
        encodings = _get_html_media_encodings(b'\n        <?xml version="1.0" encoding="utf8"?>\n        <html>\n        <head><meta charset="UTF-8">\n        </head>\n        </html>\n        ', 'text/html; charset="UTF_8"')
        self.assertEqual(list(encodings), ['utf-8', 'cp1252'])

    def test_unknown_invalid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'A character encoding should be ignored if it is unknown or invalid.'
        encodings = _get_html_media_encodings(b'\n        <html>\n        <head><meta charset="invalid">\n        </head>\n        </html>\n        ', 'text/html; charset="invalid"')
        self.assertEqual(list(encodings), ['utf-8', 'cp1252'])