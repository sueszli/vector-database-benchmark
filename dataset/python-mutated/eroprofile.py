import re
from .common import InfoExtractor
from ..compat import compat_urllib_parse_urlencode
from ..utils import ExtractorError, merge_dicts

class EroProfileIE(InfoExtractor):
    _VALID_URL = 'https?://(?:www\\.)?eroprofile\\.com/m/videos/view/(?P<id>[^/]+)'
    _LOGIN_URL = 'http://www.eroprofile.com/auth/auth.php?'
    _NETRC_MACHINE = 'eroprofile'
    _TESTS = [{'url': 'http://www.eroprofile.com/m/videos/view/sexy-babe-softcore', 'md5': 'c26f351332edf23e1ea28ce9ec9de32f', 'info_dict': {'id': '3733775', 'display_id': 'sexy-babe-softcore', 'ext': 'm4v', 'title': 'sexy babe softcore', 'thumbnail': 're:https?://.*\\.jpg', 'age_limit': 18}, 'skip': 'Video not found'}, {'url': 'http://www.eroprofile.com/m/videos/view/Try-It-On-Pee_cut_2-wmv-4shared-com-file-sharing-download-movie-file', 'md5': '1baa9602ede46ce904c431f5418d8916', 'info_dict': {'id': '1133519', 'ext': 'm4v', 'title': 'Try It On Pee_cut_2.wmv - 4shared.com - file sharing - download movie file', 'thumbnail': 're:https?://.*\\.jpg', 'age_limit': 18}, 'skip': 'Requires login'}]

    def _perform_login(self, username, password):
        if False:
            return 10
        query = compat_urllib_parse_urlencode({'username': username, 'password': password, 'url': 'http://www.eroprofile.com/'})
        login_url = self._LOGIN_URL + query
        login_page = self._download_webpage(login_url, None, False)
        m = re.search('Your username or password was incorrect\\.', login_page)
        if m:
            raise ExtractorError('Wrong username and/or password.', expected=True)
        self.report_login()
        redirect_url = self._search_regex('<script[^>]+?src="([^"]+)"', login_page, 'login redirect url')
        self._download_webpage(redirect_url, None, False)

    def _real_extract(self, url):
        if False:
            i = 10
            return i + 15
        display_id = self._match_id(url)
        webpage = self._download_webpage(url, display_id)
        m = re.search('You must be logged in to view this video\\.', webpage)
        if m:
            self.raise_login_required('This video requires login')
        video_id = self._search_regex(["glbUpdViews\\s*\\('\\d*','(\\d+)'", 'p/report/video/(\\d+)'], webpage, 'video id', default=None)
        title = self._html_search_regex(('Title:</th><td>([^<]+)</td>', '<h1[^>]*>(.+?)</h1>'), webpage, 'title')
        info = self._parse_html5_media_entries(url, webpage, video_id)[0]
        return merge_dicts(info, {'id': video_id, 'display_id': display_id, 'title': title, 'age_limit': 18})

class EroProfileAlbumIE(InfoExtractor):
    _VALID_URL = 'https?://(?:www\\.)?eroprofile\\.com/m/videos/album/(?P<id>[^/]+)'
    IE_NAME = 'EroProfile:album'
    _TESTS = [{'url': 'https://www.eroprofile.com/m/videos/album/BBW-2-893', 'info_dict': {'id': 'BBW-2-893', 'title': 'BBW 2'}, 'playlist_mincount': 486}]

    def _extract_from_page(self, page):
        if False:
            for i in range(10):
                print('nop')
        for url in re.findall('href=".*?(/m/videos/view/[^"]+)"', page):
            yield self.url_result(f'https://www.eroprofile.com{url}', EroProfileIE.ie_key())

    def _entries(self, playlist_id, first_page):
        if False:
            return 10
        yield from self._extract_from_page(first_page)
        page_urls = re.findall(f'href=".*?(/m/videos/album/{playlist_id}\\?pnum=(\\d+))"', first_page)
        max_page = max((int(n) for (_, n) in page_urls))
        for n in range(2, max_page + 1):
            url = f'https://www.eroprofile.com/m/videos/album/{playlist_id}?pnum={n}'
            yield from self._extract_from_page(self._download_webpage(url, playlist_id, note=f'Downloading playlist page {int(n) - 1}'))

    def _real_extract(self, url):
        if False:
            print('Hello World!')
        playlist_id = self._match_id(url)
        first_page = self._download_webpage(url, playlist_id, note='Downloading playlist')
        playlist_title = self._search_regex('<title>Album: (.*) - EroProfile</title>', first_page, 'playlist_title')
        return self.playlist_result(self._entries(playlist_id, first_page), playlist_id, playlist_title)