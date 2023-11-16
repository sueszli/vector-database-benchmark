from ..utils import str_to_int
from .keezmovies import KeezMoviesIE

class ExtremeTubeIE(KeezMoviesIE):
    _VALID_URL = 'https?://(?:www\\.)?extremetube\\.com/(?:[^/]+/)?video/(?P<id>[^/#?&]+)'
    _TESTS = [{'url': 'http://www.extremetube.com/video/music-video-14-british-euro-brit-european-cumshots-swallow-652431', 'md5': '92feaafa4b58e82f261e5419f39c60cb', 'info_dict': {'id': 'music-video-14-british-euro-brit-european-cumshots-swallow-652431', 'ext': 'mp4', 'title': 'Music Video 14 british euro brit european cumshots swallow', 'uploader': 'anonim', 'view_count': int, 'age_limit': 18}}, {'url': 'http://www.extremetube.com/gay/video/abcde-1234', 'only_matching': True}, {'url': 'http://www.extremetube.com/video/latina-slut-fucked-by-fat-black-dick', 'only_matching': True}, {'url': 'http://www.extremetube.com/video/652431', 'only_matching': True}]

    def _real_extract(self, url):
        if False:
            for i in range(10):
                print('nop')
        (webpage, info) = self._extract_info(url)
        if not info['title']:
            info['title'] = self._search_regex('<h1[^>]+title="([^"]+)"[^>]*>', webpage, 'title')
        uploader = self._html_search_regex('Uploaded by:\\s*</[^>]+>\\s*<a[^>]+>(.+?)</a>', webpage, 'uploader', fatal=False)
        view_count = str_to_int(self._search_regex('Views:\\s*</[^>]+>\\s*<[^>]+>([\\d,\\.]+)</', webpage, 'view count', fatal=False))
        info.update({'uploader': uploader, 'view_count': view_count})
        return info