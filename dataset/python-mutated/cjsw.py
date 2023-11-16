from .common import InfoExtractor
from ..utils import determine_ext, unescapeHTML

class CJSWIE(InfoExtractor):
    _VALID_URL = 'https?://(?:www\\.)?cjsw\\.com/program/(?P<program>[^/]+)/episode/(?P<id>\\d+)'
    _TESTS = [{'url': 'http://cjsw.com/program/freshly-squeezed/episode/20170620', 'md5': 'cee14d40f1e9433632c56e3d14977120', 'info_dict': {'id': '91d9f016-a2e7-46c5-8dcb-7cbcd7437c41', 'ext': 'mp3', 'title': 'Freshly Squeezed – Episode June 20, 2017', 'description': 'md5:c967d63366c3898a80d0c7b0ff337202', 'series': 'Freshly Squeezed', 'episode_id': '20170620'}}, {'url': 'http://cjsw.com/program/road-pops/episode/20170707/', 'only_matching': True}]

    def _real_extract(self, url):
        if False:
            for i in range(10):
                print('nop')
        mobj = self._match_valid_url(url)
        (program, episode_id) = mobj.group('program', 'id')
        audio_id = '%s/%s' % (program, episode_id)
        webpage = self._download_webpage(url, episode_id)
        title = unescapeHTML(self._search_regex(('<h1[^>]+class=["\\\']episode-header__title["\\\'][^>]*>(?P<title>[^<]+)', 'data-audio-title=(["\\\'])(?P<title>(?:(?!\\1).)+)\\1'), webpage, 'title', group='title'))
        audio_url = self._search_regex('<button[^>]+data-audio-src=(["\\\'])(?P<url>(?:(?!\\1).)+)\\1', webpage, 'audio url', group='url')
        audio_id = self._search_regex('/([\\da-f]{8}-[\\da-f]{4}-[\\da-f]{4}-[\\da-f]{4}-[\\da-f]{12})\\.mp3', audio_url, 'audio id', default=audio_id)
        formats = [{'url': audio_url, 'ext': determine_ext(audio_url, 'mp3'), 'vcodec': 'none'}]
        description = self._html_search_regex('<p>(?P<description>.+?)</p>', webpage, 'description', default=None)
        series = self._search_regex('data-showname=(["\\\'])(?P<name>(?:(?!\\1).)+)\\1', webpage, 'series', default=program, group='name')
        return {'id': audio_id, 'title': title, 'description': description, 'formats': formats, 'series': series, 'episode_id': episode_id}