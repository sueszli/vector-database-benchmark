import re
from .common import InfoExtractor

class AcademicEarthCourseIE(InfoExtractor):
    _VALID_URL = '^https?://(?:www\\.)?academicearth\\.org/playlists/(?P<id>[^?#/]+)'
    IE_NAME = 'AcademicEarth:Course'
    _TEST = {'url': 'http://academicearth.org/playlists/laws-of-nature/', 'info_dict': {'id': 'laws-of-nature', 'title': 'Laws of Nature', 'description': 'Introduce yourself to the laws of nature with these free online college lectures from Yale, Harvard, and MIT.'}, 'playlist_count': 3}

    def _real_extract(self, url):
        if False:
            for i in range(10):
                print('nop')
        playlist_id = self._match_id(url)
        webpage = self._download_webpage(url, playlist_id)
        title = self._html_search_regex('<h1 class="playlist-name"[^>]*?>(.*?)</h1>', webpage, 'title')
        description = self._html_search_regex('<p class="excerpt"[^>]*?>(.*?)</p>', webpage, 'description', fatal=False)
        urls = re.findall('<li class="lecture-preview">\\s*?<a target="_blank" href="([^"]+)">', webpage)
        entries = [self.url_result(u) for u in urls]
        return {'_type': 'playlist', 'id': playlist_id, 'title': title, 'description': description, 'entries': entries}