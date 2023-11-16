import os
import sys
import unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import collections
from test.helper import gettestcases
from yt_dlp.extractor import FacebookIE, YoutubeIE, gen_extractors

class TestAllURLsMatching(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.ies = gen_extractors()

    def matching_ies(self, url):
        if False:
            for i in range(10):
                print('nop')
        return [ie.IE_NAME for ie in self.ies if ie.suitable(url) and ie.IE_NAME != 'generic']

    def assertMatch(self, url, ie_list):
        if False:
            return 10
        self.assertEqual(self.matching_ies(url), ie_list)

    def test_youtube_playlist_matching(self):
        if False:
            return 10
        assertPlaylist = lambda url: self.assertMatch(url, ['youtube:playlist'])
        assertTab = lambda url: self.assertMatch(url, ['youtube:tab'])
        assertPlaylist('ECUl4u3cNGP61MdtwGTqZA0MreSaDybji8')
        assertPlaylist('UUBABnxM4Ar9ten8Mdjj1j0Q')
        assertPlaylist('PL63F0C78739B09958')
        assertTab('https://www.youtube.com/AsapSCIENCE')
        assertTab('https://www.youtube.com/embedded')
        assertTab('https://www.youtube.com/playlist?list=UUBABnxM4Ar9ten8Mdjj1j0Q')
        assertTab('https://www.youtube.com/playlist?list=PLwP_SiAcdui0KVebT0mU9Apz359a4ubsC')
        assertTab('https://www.youtube.com/watch?v=AV6J6_AeFEQ&playnext=1&list=PL4023E734DA416012')
        self.assertFalse('youtube:playlist' in self.matching_ies('PLtS2H6bU1M'))
        assertTab('https://www.youtube.com/playlist?list=MCUS.20142101')

    def test_youtube_matching(self):
        if False:
            while True:
                i = 10
        self.assertTrue(YoutubeIE.suitable('PLtS2H6bU1M'))
        self.assertFalse(YoutubeIE.suitable('https://www.youtube.com/watch?v=AV6J6_AeFEQ&playnext=1&list=PL4023E734DA416012'))
        self.assertMatch('http://youtu.be/BaW_jenozKc', ['youtube'])
        self.assertMatch('https://youtube.googleapis.com/v/BaW_jenozKc', ['youtube'])
        self.assertMatch('http://www.cleanvideosearch.com/media/action/yt/watch?videoId=8v_4O44sfjM', ['youtube'])

    def test_youtube_channel_matching(self):
        if False:
            i = 10
            return i + 15
        assertChannel = lambda url: self.assertMatch(url, ['youtube:tab'])
        assertChannel('https://www.youtube.com/channel/HCtnHdj3df7iM')
        assertChannel('https://www.youtube.com/channel/HCtnHdj3df7iM?feature=gb_ch_rec')
        assertChannel('https://www.youtube.com/channel/HCtnHdj3df7iM/videos')

    def test_youtube_user_matching(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMatch('http://www.youtube.com/NASAgovVideo/videos', ['youtube:tab'])

    def test_youtube_feeds(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMatch('https://www.youtube.com/feed/library', ['youtube:tab'])
        self.assertMatch('https://www.youtube.com/feed/history', ['youtube:tab'])
        self.assertMatch('https://www.youtube.com/feed/watch_later', ['youtube:tab'])
        self.assertMatch('https://www.youtube.com/feed/subscriptions', ['youtube:tab'])

    def test_youtube_search_matching(self):
        if False:
            i = 10
            return i + 15
        self.assertMatch('http://www.youtube.com/results?search_query=making+mustard', ['youtube:search_url'])
        self.assertMatch('https://www.youtube.com/results?baz=bar&search_query=youtube-dl+test+video&filters=video&lclk=video', ['youtube:search_url'])

    def test_facebook_matching(self):
        if False:
            print('Hello World!')
        self.assertTrue(FacebookIE.suitable('https://www.facebook.com/Shiniknoh#!/photo.php?v=10153317450565268'))
        self.assertTrue(FacebookIE.suitable('https://www.facebook.com/cindyweather?fref=ts#!/photo.php?v=10152183998945793'))

    def test_no_duplicates(self):
        if False:
            print('Hello World!')
        ies = gen_extractors()
        for tc in gettestcases(include_onlymatching=True):
            url = tc['url']
            for ie in ies:
                if type(ie).__name__ in ('GenericIE', tc['name'] + 'IE'):
                    self.assertTrue(ie.suitable(url), f'{type(ie).__name__} should match URL {url!r}')
                else:
                    self.assertFalse(ie.suitable(url), f"{type(ie).__name__} should not match URL {url!r} . That URL belongs to {tc['name']}.")

    def test_keywords(self):
        if False:
            return 10
        self.assertMatch(':ytsubs', ['youtube:subscriptions'])
        self.assertMatch(':ytsubscriptions', ['youtube:subscriptions'])
        self.assertMatch(':ythistory', ['youtube:history'])

    def test_vimeo_matching(self):
        if False:
            print('Hello World!')
        self.assertMatch('https://vimeo.com/channels/tributes', ['vimeo:channel'])
        self.assertMatch('https://vimeo.com/channels/31259', ['vimeo:channel'])
        self.assertMatch('https://vimeo.com/channels/31259/53576664', ['vimeo'])
        self.assertMatch('https://vimeo.com/user7108434', ['vimeo:user'])
        self.assertMatch('https://vimeo.com/user7108434/videos', ['vimeo:user'])
        self.assertMatch('https://vimeo.com/user21297594/review/75524534/3c257a1b5d', ['vimeo:review'])

    def test_soundcloud_not_matching_sets(self):
        if False:
            while True:
                i = 10
        self.assertMatch('http://soundcloud.com/floex/sets/gone-ep', ['soundcloud:set'])

    def test_tumblr(self):
        if False:
            i = 10
            return i + 15
        self.assertMatch('http://tatianamaslanydaily.tumblr.com/post/54196191430/orphan-black-dvd-extra-behind-the-scenes', ['Tumblr'])
        self.assertMatch('http://tatianamaslanydaily.tumblr.com/post/54196191430', ['Tumblr'])

    def test_pbs(self):
        if False:
            print('Hello World!')
        self.assertMatch('http://video.pbs.org/viralplayer/2365173446/', ['pbs'])
        self.assertMatch('http://video.pbs.org/widget/partnerplayer/980042464/', ['pbs'])

    def test_no_duplicated_ie_names(self):
        if False:
            i = 10
            return i + 15
        name_accu = collections.defaultdict(list)
        for ie in self.ies:
            name_accu[ie.IE_NAME.lower()].append(type(ie).__name__)
        for (ie_name, ie_list) in name_accu.items():
            self.assertEqual(len(ie_list), 1, f'''Multiple extractors with the same IE_NAME "{ie_name}" ({', '.join(ie_list)})''')
if __name__ == '__main__':
    unittest.main()