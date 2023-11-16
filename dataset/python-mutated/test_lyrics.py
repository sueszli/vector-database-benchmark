"""Tests for the 'lyrics' plugin."""
import itertools
import os
import re
import unittest
from test import _common
from unittest.mock import MagicMock, patch
import confuse
import requests
from beets import logging
from beets.library import Item
from beets.util import bytestring_path
from beetsplug import lyrics
log = logging.getLogger('beets.test_lyrics')
raw_backend = lyrics.Backend({}, log)
google = lyrics.Google(MagicMock(), log)
genius = lyrics.Genius(MagicMock(), log)
tekstowo = lyrics.Tekstowo(MagicMock(), log)
lrclib = lyrics.LRCLib(MagicMock(), log)

class LyricsPluginTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        'Set up configuration.'
        lyrics.LyricsPlugin()

    def test_search_artist(self):
        if False:
            print('Hello World!')
        item = Item(artist='Alice ft. Bob', title='song')
        self.assertIn(('Alice ft. Bob', ['song']), lyrics.search_pairs(item))
        self.assertIn(('Alice', ['song']), lyrics.search_pairs(item))
        item = Item(artist='Alice feat Bob', title='song')
        self.assertIn(('Alice feat Bob', ['song']), lyrics.search_pairs(item))
        self.assertIn(('Alice', ['song']), lyrics.search_pairs(item))
        item = Item(artist='Alice feat. Bob', title='song')
        self.assertIn(('Alice feat. Bob', ['song']), lyrics.search_pairs(item))
        self.assertIn(('Alice', ['song']), lyrics.search_pairs(item))
        item = Item(artist='Alice feats Bob', title='song')
        self.assertIn(('Alice feats Bob', ['song']), lyrics.search_pairs(item))
        self.assertNotIn(('Alice', ['song']), lyrics.search_pairs(item))
        item = Item(artist='Alice featuring Bob', title='song')
        self.assertIn(('Alice featuring Bob', ['song']), lyrics.search_pairs(item))
        self.assertIn(('Alice', ['song']), lyrics.search_pairs(item))
        item = Item(artist='Alice & Bob', title='song')
        self.assertIn(('Alice & Bob', ['song']), lyrics.search_pairs(item))
        self.assertIn(('Alice', ['song']), lyrics.search_pairs(item))
        item = Item(artist='Alice and Bob', title='song')
        self.assertIn(('Alice and Bob', ['song']), lyrics.search_pairs(item))
        self.assertIn(('Alice', ['song']), lyrics.search_pairs(item))
        item = Item(artist='Alice and Bob', title='song')
        self.assertEqual(('Alice and Bob', ['song']), list(lyrics.search_pairs(item))[0])

    def test_search_artist_sort(self):
        if False:
            i = 10
            return i + 15
        item = Item(artist='CHVRCHΞS', title='song', artist_sort='CHVRCHES')
        self.assertIn(('CHVRCHΞS', ['song']), lyrics.search_pairs(item))
        self.assertIn(('CHVRCHES', ['song']), lyrics.search_pairs(item))
        self.assertEqual(('CHVRCHΞS', ['song']), list(lyrics.search_pairs(item))[0])
        item = Item(artist='横山克', title='song', artist_sort='Masaru Yokoyama')
        self.assertIn(('横山克', ['song']), lyrics.search_pairs(item))
        self.assertIn(('Masaru Yokoyama', ['song']), lyrics.search_pairs(item))
        self.assertEqual(('横山克', ['song']), list(lyrics.search_pairs(item))[0])

    def test_search_pairs_multi_titles(self):
        if False:
            return 10
        item = Item(title='1 / 2', artist='A')
        self.assertIn(('A', ['1 / 2']), lyrics.search_pairs(item))
        self.assertIn(('A', ['1', '2']), lyrics.search_pairs(item))
        item = Item(title='1/2', artist='A')
        self.assertIn(('A', ['1/2']), lyrics.search_pairs(item))
        self.assertIn(('A', ['1', '2']), lyrics.search_pairs(item))

    def test_search_pairs_titles(self):
        if False:
            while True:
                i = 10
        item = Item(title='Song (live)', artist='A')
        self.assertIn(('A', ['Song']), lyrics.search_pairs(item))
        self.assertIn(('A', ['Song (live)']), lyrics.search_pairs(item))
        item = Item(title='Song (live) (new)', artist='A')
        self.assertIn(('A', ['Song']), lyrics.search_pairs(item))
        self.assertIn(('A', ['Song (live) (new)']), lyrics.search_pairs(item))
        item = Item(title='Song (live (new))', artist='A')
        self.assertIn(('A', ['Song']), lyrics.search_pairs(item))
        self.assertIn(('A', ['Song (live (new))']), lyrics.search_pairs(item))
        item = Item(title='Song ft. B', artist='A')
        self.assertIn(('A', ['Song']), lyrics.search_pairs(item))
        self.assertIn(('A', ['Song ft. B']), lyrics.search_pairs(item))
        item = Item(title='Song featuring B', artist='A')
        self.assertIn(('A', ['Song']), lyrics.search_pairs(item))
        self.assertIn(('A', ['Song featuring B']), lyrics.search_pairs(item))
        item = Item(title='Song and B', artist='A')
        self.assertNotIn(('A', ['Song']), lyrics.search_pairs(item))
        self.assertIn(('A', ['Song and B']), lyrics.search_pairs(item))
        item = Item(title='Song: B', artist='A')
        self.assertIn(('A', ['Song']), lyrics.search_pairs(item))
        self.assertIn(('A', ['Song: B']), lyrics.search_pairs(item))

    def test_remove_credits(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(lyrics.remove_credits("It's close to midnight\n                                     Lyrics brought by example.com"), "It's close to midnight")
        self.assertEqual(lyrics.remove_credits('Lyrics brought by example.com'), '')
        text = "Look at all the shit that i done bought her\n                  See lyrics ain't nothin\n                  if the beat aint crackin"
        self.assertEqual(lyrics.remove_credits(text), text)

    def test_is_lyrics(self):
        if False:
            print('Hello World!')
        texts = ['LyricsMania.com - Copyright (c) 2013 - All Rights Reserved']
        texts += ['All material found on this site is property\n\n                     of mywickedsongtext brand']
        for t in texts:
            self.assertFalse(google.is_lyrics(t))

    def test_slugify(self):
        if False:
            i = 10
            return i + 15
        text = 'http://site.com/çafe-au_lait(boisson)'
        self.assertEqual(google.slugify(text), 'http://site.com/cafe_au_lait')

    def test_scrape_strip_cruft(self):
        if False:
            return 10
        text = "<!--lyrics below-->\n                  &nbsp;one\n                  <br class='myclass'>\n                  two  !\n                  <br><br \\>\n                  <blink>four</blink>"
        self.assertEqual(lyrics._scrape_strip_cruft(text, True), 'one\ntwo !\n\nfour')

    def test_scrape_strip_scripts(self):
        if False:
            for i in range(10):
                print('nop')
        text = 'foo<script>bar</script>baz'
        self.assertEqual(lyrics._scrape_strip_cruft(text, True), 'foobaz')

    def test_scrape_strip_tag_in_comment(self):
        if False:
            print('Hello World!')
        text = 'foo<!--<bar>-->qux'
        self.assertEqual(lyrics._scrape_strip_cruft(text, True), 'fooqux')

    def test_scrape_merge_paragraphs(self):
        if False:
            print('Hello World!')
        text = "one</p>   <p class='myclass'>two</p><p>three"
        self.assertEqual(lyrics._scrape_merge_paragraphs(text), 'one\ntwo\nthree')

    def test_missing_lyrics(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(google.is_lyrics(LYRICS_TEXTS['missing_texts']))

def url_to_filename(url):
    if False:
        return 10
    url = re.sub('https?://|www.', '', url)
    url = re.sub('.html', '', url)
    fn = ''.join((x for x in url if x.isalnum() or x == '/'))
    fn = fn.split('/')
    fn = os.path.join(LYRICS_ROOT_DIR, bytestring_path(fn[0]), bytestring_path(fn[-1] + '.txt'))
    return fn

class MockFetchUrl:

    def __init__(self, pathval='fetched_path'):
        if False:
            return 10
        self.pathval = pathval
        self.fetched = None

    def __call__(self, url, filename=None):
        if False:
            return 10
        self.fetched = url
        fn = url_to_filename(url)
        with open(fn, encoding='utf8') as f:
            content = f.read()
        return content

class LyricsAssertions:
    """A mixin with lyrics-specific assertions."""

    def assertLyricsContentOk(self, title, text, msg=''):
        if False:
            i = 10
            return i + 15
        'Compare lyrics text to expected lyrics for given title.'
        if not text:
            return
        keywords = set(LYRICS_TEXTS[google.slugify(title)].split())
        words = {x.strip('.?, ()') for x in text.lower().split()}
        if not keywords <= words:
            details = f'{keywords!r} is not a subset of {words!r}. Words only in expected set {keywords - words!r}, Words only in result set {words - keywords!r}.'
            self.fail(f'{details} : {msg}')
LYRICS_ROOT_DIR = os.path.join(_common.RSRC, b'lyrics')
yaml_path = os.path.join(_common.RSRC, b'lyricstext.yaml')
LYRICS_TEXTS = confuse.load_yaml(yaml_path)

class LyricsGoogleBaseTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Set up configuration.'
        try:
            __import__('bs4')
        except ImportError:
            self.skipTest('Beautiful Soup 4 not available')

class LyricsPluginSourcesTest(LyricsGoogleBaseTest, LyricsAssertions):
    """Check that beets google custom search engine sources are correctly
    scraped.
    """
    DEFAULT_SONG = dict(artist='The Beatles', title='Lady Madonna')
    DEFAULT_SOURCES = [dict(DEFAULT_SONG, backend=lyrics.Genius, skip=os.environ.get('GITHUB_ACTIONS') == 'true'), dict(artist='Boy In Space', title='u n eye', backend=lyrics.Tekstowo)]
    GOOGLE_SOURCES = [dict(DEFAULT_SONG, url='http://www.absolutelyrics.com', path='/lyrics/view/the_beatles/lady_madonna'), dict(DEFAULT_SONG, url='http://www.azlyrics.com', path='/lyrics/beatles/ladymadonna.html', skip=os.environ.get('GITHUB_ACTIONS') == 'true'), dict(DEFAULT_SONG, url='http://www.chartlyrics.com', path='/_LsLsZ7P4EK-F-LD4dJgDQ/Lady+Madonna.aspx'), dict(url='http://www.lacoccinelle.net', artist='Jacques Brel', title='Amsterdam', path='/paroles-officielles/275679.html'), dict(DEFAULT_SONG, url='http://letras.mus.br/', path='the-beatles/275/'), dict(DEFAULT_SONG, url='http://www.lyricsmania.com/', path='lady_madonna_lyrics_the_beatles.html'), dict(DEFAULT_SONG, url='http://www.lyricsmode.com', path='/lyrics/b/beatles/lady_madonna.html'), dict(url='http://www.lyricsontop.com', artist='Amy Winehouse', title="Jazz'n'blues", path='/amy-winehouse-songs/jazz-n-blues-lyrics.html'), dict(url='http://www.paroles.net/', artist='Lilly Wood & the prick', title="Hey it's ok", path='lilly-wood-the-prick/paroles-hey-it-s-ok'), dict(DEFAULT_SONG, url='http://www.songlyrics.com', path='/the-beatles/lady-madonna-lyrics'), dict(DEFAULT_SONG, url='http://www.sweetslyrics.com', path='/761696.The%20Beatles%20-%20Lady%20Madonna.html')]

    def setUp(self):
        if False:
            while True:
                i = 10
        LyricsGoogleBaseTest.setUp(self)
        self.plugin = lyrics.LyricsPlugin()

    @unittest.skipUnless(os.environ.get('INTEGRATION_TEST', '0') == '1', 'integration testing not enabled')
    def test_backend_sources_ok(self):
        if False:
            i = 10
            return i + 15
        'Test default backends with songs known to exist in respective\n        databases.\n        '
        sources = [s for s in self.DEFAULT_SOURCES if not s.get('skip', False)]
        for s in sources:
            with self.subTest(s['backend'].__name__):
                backend = s['backend'](self.plugin.config, self.plugin._log)
                res = backend.fetch(s['artist'], s['title'])
                self.assertLyricsContentOk(s['title'], res)

    @unittest.skipUnless(os.environ.get('INTEGRATION_TEST', '0') == '1', 'integration testing not enabled')
    def test_google_sources_ok(self):
        if False:
            print('Hello World!')
        'Test if lyrics present on websites registered in beets google custom\n        search engine are correctly scraped.\n        '
        sources = [s for s in self.GOOGLE_SOURCES if not s.get('skip', False)]
        for s in sources:
            url = s['url'] + s['path']
            res = lyrics.scrape_lyrics_from_html(raw_backend.fetch_url(url))
            self.assertTrue(google.is_lyrics(res), url)
            self.assertLyricsContentOk(s['title'], res, url)

class LyricsGooglePluginMachineryTest(LyricsGoogleBaseTest, LyricsAssertions):
    """Test scraping heuristics on a fake html page."""
    source = dict(url='http://www.example.com', artist='John Doe', title='Beets song', path='/lyrics/beetssong')

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Set up configuration'
        LyricsGoogleBaseTest.setUp(self)
        self.plugin = lyrics.LyricsPlugin()

    @patch.object(lyrics.Backend, 'fetch_url', MockFetchUrl())
    def test_mocked_source_ok(self):
        if False:
            while True:
                i = 10
        'Test that lyrics of the mocked page are correctly scraped'
        url = self.source['url'] + self.source['path']
        res = lyrics.scrape_lyrics_from_html(raw_backend.fetch_url(url))
        self.assertTrue(google.is_lyrics(res), url)
        self.assertLyricsContentOk(self.source['title'], res, url)

    @patch.object(lyrics.Backend, 'fetch_url', MockFetchUrl())
    def test_is_page_candidate_exact_match(self):
        if False:
            for i in range(10):
                print('nop')
        'Test matching html page title with song infos -- when song infos are\n        present in the title.\n        '
        from bs4 import BeautifulSoup, SoupStrainer
        s = self.source
        url = str(s['url'] + s['path'])
        html = raw_backend.fetch_url(url)
        soup = BeautifulSoup(html, 'html.parser', parse_only=SoupStrainer('title'))
        self.assertEqual(google.is_page_candidate(url, soup.title.string, s['title'], s['artist']), True, url)

    def test_is_page_candidate_fuzzy_match(self):
        if False:
            print('Hello World!')
        'Test matching html page title with song infos -- when song infos are\n        not present in the title.\n        '
        s = self.source
        url = s['url'] + s['path']
        url_title = 'example.com | Beats song by John doe'
        self.assertEqual(google.is_page_candidate(url, url_title, s['title'], s['artist']), True, url)
        url_title = 'example.com | seets bong lyrics by John doe'
        self.assertEqual(google.is_page_candidate(url, url_title, s['title'], s['artist']), False, url)

    def test_is_page_candidate_special_chars(self):
        if False:
            print('Hello World!')
        "Ensure that `is_page_candidate` doesn't crash when the artist\n        and such contain special regular expression characters.\n        "
        s = self.source
        url = s['url'] + s['path']
        url_title = 'foo'
        google.is_page_candidate(url, url_title, s['title'], 'Sunn O)))')

class GeniusBaseTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        'Set up configuration.'
        try:
            __import__('bs4')
        except ImportError:
            self.skipTest('Beautiful Soup 4 not available')

class GeniusScrapeLyricsFromHtmlTest(GeniusBaseTest):
    """tests Genius._scrape_lyrics_from_html()"""

    def setUp(self):
        if False:
            while True:
                i = 10
        'Set up configuration'
        GeniusBaseTest.setUp(self)
        self.plugin = lyrics.LyricsPlugin()

    def test_no_lyrics_div(self):
        if False:
            print('Hello World!')
        'Ensure we don\'t crash when the scraping the html for a genius page\n        doesn\'t contain <div class="lyrics"></div>\n        '
        url = 'https://genius.com/sample'
        mock = MockFetchUrl()
        self.assertEqual(genius._scrape_lyrics_from_html(mock(url)), None)

    def test_good_lyrics(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure we are able to scrape a page with lyrics'
        url = 'https://genius.com/Ttng-chinchilla-lyrics'
        mock = MockFetchUrl()
        self.assertIsNotNone(genius._scrape_lyrics_from_html(mock(url)))

class GeniusFetchTest(GeniusBaseTest):
    """tests Genius.fetch()"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Set up configuration'
        GeniusBaseTest.setUp(self)
        self.plugin = lyrics.LyricsPlugin()

    @patch.object(lyrics.Genius, '_scrape_lyrics_from_html')
    @patch.object(lyrics.Backend, 'fetch_url', return_value=True)
    def test_json(self, mock_fetch_url, mock_scrape):
        if False:
            for i in range(10):
                print('nop')
        "Ensure we're finding artist matches"
        with patch.object(lyrics.Genius, '_search', return_value={'response': {'hits': [{'result': {'primary_artist': {'name': '\u200bblackbear'}, 'url': 'blackbear_url'}}, {'result': {'primary_artist': {'name': 'El-p'}, 'url': 'El-p_url'}}]}}) as mock_json:
            self.assertIsNotNone(genius.fetch('blackbear', 'Idfc'))
            mock_fetch_url.assert_called_once_with('blackbear_url')
            mock_scrape.assert_called_once_with(True)
            self.assertIsNotNone(genius.fetch('El-p', 'Idfc'))
            mock_fetch_url.assert_called_with('El-p_url')
            mock_scrape.assert_called_with(True)
            self.assertIsNone(genius.fetch('doesntexist', 'none'))
            mock_json.return_value = None
            self.assertIsNone(genius.fetch('blackbear', 'Idfc'))

class TekstowoBaseTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Set up configuration.'
        try:
            __import__('bs4')
        except ImportError:
            self.skipTest('Beautiful Soup 4 not available')

class TekstowoExtractLyricsTest(TekstowoBaseTest):
    """tests Tekstowo.extract_lyrics()"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Set up configuration'
        TekstowoBaseTest.setUp(self)
        self.plugin = lyrics.LyricsPlugin()
        tekstowo.config = self.plugin.config

    def test_good_lyrics(self):
        if False:
            return 10
        'Ensure we are able to scrape a page with lyrics'
        url = 'https://www.tekstowo.pl/piosenka,24kgoldn,city_of_angels_1.html'
        mock = MockFetchUrl()
        self.assertIsNotNone(tekstowo.extract_lyrics(mock(url), '24kGoldn', 'City of Angels'))

    def test_no_lyrics(self):
        if False:
            for i in range(10):
                print('nop')
        "Ensure we don't crash when the scraping the html for a Tekstowo page\n        doesn't contain lyrics\n        "
        url = 'https://www.tekstowo.pl/piosenka,beethoven,beethoven_piano_sonata_17_tempest_the_3rd_movement.html'
        mock = MockFetchUrl()
        self.assertEqual(tekstowo.extract_lyrics(mock(url), 'Beethoven', 'Beethoven Piano Sonata 17Tempest The 3rd Movement'), None)

    def test_song_no_match(self):
        if False:
            i = 10
            return i + 15
        'Ensure we return None when a song does not match the search query'
        url = 'https://www.tekstowo.pl/piosenka,bailey_bigger,black_eyed_susan.html'
        mock = MockFetchUrl()
        self.assertEqual(tekstowo.extract_lyrics(mock(url), 'Kelly Bailey', 'Black Mesa Inbound'), None)

class TekstowoParseSearchResultsTest(TekstowoBaseTest):
    """tests Tekstowo.parse_search_results()"""

    def setUp(self):
        if False:
            return 10
        'Set up configuration'
        TekstowoBaseTest.setUp(self)
        self.plugin = lyrics.LyricsPlugin()

    def test_multiple_results(self):
        if False:
            i = 10
            return i + 15
        'Ensure we are able to scrape a page with multiple search results'
        url = 'https://www.tekstowo.pl/szukaj,wykonawca,juice+wrld,tytul,lucid+dreams.html'
        mock = MockFetchUrl()
        self.assertEqual(tekstowo.parse_search_results(mock(url)), 'http://www.tekstowo.pl/piosenka,juice_wrld,lucid_dreams__remix__ft__lil_uzi_vert.html')

    def test_no_results(self):
        if False:
            while True:
                i = 10
        'Ensure we are able to scrape a page with no search results'
        url = 'https://www.tekstowo.pl/szukaj,wykonawca,agfdgja,tytul,agfdgafg.html'
        mock = MockFetchUrl()
        self.assertEqual(tekstowo.parse_search_results(mock(url)), None)

class TekstowoIntegrationTest(TekstowoBaseTest, LyricsAssertions):
    """Tests Tekstowo lyric source with real requests"""

    def setUp(self):
        if False:
            print('Hello World!')
        'Set up configuration'
        TekstowoBaseTest.setUp(self)
        self.plugin = lyrics.LyricsPlugin()
        tekstowo.config = self.plugin.config

    @unittest.skipUnless(os.environ.get('INTEGRATION_TEST', '0') == '1', 'integration testing not enabled')
    def test_normal(self):
        if False:
            i = 10
            return i + 15
        "Ensure we can fetch a song's lyrics in the ordinary case"
        lyrics = tekstowo.fetch('Boy in Space', 'u n eye')
        self.assertLyricsContentOk('u n eye', lyrics)

    @unittest.skipUnless(os.environ.get('INTEGRATION_TEST', '0') == '1', 'integration testing not enabled')
    def test_no_matching_results(self):
        if False:
            return 10
        'Ensure we fetch nothing if there are search results\n        returned but no matches'
        lyrics = tekstowo.fetch('Kelly Bailey', 'Black Mesa Inbound')
        self.assertEqual(lyrics, None)

class LRCLibLyricsTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.plugin = lyrics.LyricsPlugin()
        lrclib.config = self.plugin.config

    @patch('beetsplug.lyrics.requests.get')
    def test_fetch_synced_lyrics(self, mock_get):
        if False:
            print('Hello World!')
        mock_response = {'syncedLyrics': '[00:00.00] la la la', 'plainLyrics': 'la la la'}
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200
        lyrics = lrclib.fetch('la', 'la', 'la', 999)
        self.assertEqual(lyrics, mock_response['plainLyrics'])
        self.plugin.config['synced'] = True
        lyrics = lrclib.fetch('la', 'la', 'la', 999)
        self.assertEqual(lyrics, mock_response['syncedLyrics'])

    @patch('beetsplug.lyrics.requests.get')
    def test_fetch_plain_lyrics(self, mock_get):
        if False:
            for i in range(10):
                print('nop')
        mock_response = {'syncedLyrics': '', 'plainLyrics': 'la la la'}
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200
        lyrics = lrclib.fetch('la', 'la', 'la', 999)
        self.assertEqual(lyrics, mock_response['plainLyrics'])

    @patch('beetsplug.lyrics.requests.get')
    def test_fetch_not_found(self, mock_get):
        if False:
            return 10
        mock_response = {'statusCode': 404, 'error': 'Not Found', 'message': 'Failed to find specified track'}
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 404
        lyrics = lrclib.fetch('la', 'la', 'la', 999)
        self.assertIsNone(lyrics)

    @patch('beetsplug.lyrics.requests.get')
    def test_fetch_exception(self, mock_get):
        if False:
            print('Hello World!')
        mock_get.side_effect = requests.RequestException
        lyrics = lrclib.fetch('la', 'la', 'la', 999)
        self.assertIsNone(lyrics)

class LRCLibIntegrationTest(LyricsAssertions):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.plugin = lyrics.LyricsPlugin()
        lrclib.config = self.plugin.config

    @unittest.skipUnless(os.environ.get('INTEGRATION_TEST', '0') == '1', 'integration testing not enabled')
    def test_track_with_lyrics(self):
        if False:
            while True:
                i = 10
        lyrics = lrclib.fetch('Boy in Space', 'u n eye', 'Live EP', 160)
        self.assertLyricsContentOk('u n eye', lyrics)

    @unittest.skipUnless(os.environ.get('INTEGRATION_TEST', '0') == '1', 'integration testing not enabled')
    def test_instrumental_track(self):
        if False:
            while True:
                i = 10
        lyrics = lrclib.fetch('Kelly Bailey', 'Black Mesa Inbound', 'Half Life 2 Soundtrack', 134)
        self.assertIsNone(lyrics)

    @unittest.skipUnless(os.environ.get('INTEGRATION_TEST', '0') == '1', 'integration testing not enabled')
    def test_nonexistent_track(self):
        if False:
            return 10
        lyrics = lrclib.fetch('blah', 'blah', 'blah', 999)
        self.assertIsNone(lyrics)

class SlugTests(unittest.TestCase):

    def test_slug(self):
        if False:
            for i in range(10):
                print('nop')
        text = 'test'
        self.assertEqual(lyrics.slug(text), 'test')
        text = 'Mørdag'
        self.assertEqual(lyrics.slug(text), 'mordag')
        text = "l'été c'est fait pour jouer"
        self.assertEqual(lyrics.slug(text), 'l-ete-c-est-fait-pour-jouer')
        text = 'çafe au lait (boisson)'
        self.assertEqual(lyrics.slug(text), 'cafe-au-lait-boisson')
        text = 'Multiple  spaces -- and symbols! -- merged'
        self.assertEqual(lyrics.slug(text), 'multiple-spaces-and-symbols-merged')
        text = '\u200bno-width-space'
        self.assertEqual(lyrics.slug(text), 'no-width-space')
        dashes = ['\u200d', '‐']
        for (dash1, dash2) in itertools.combinations(dashes, 2):
            self.assertEqual(lyrics.slug(dash1), lyrics.slug(dash2))

def suite():
    if False:
        for i in range(10):
            print('nop')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')