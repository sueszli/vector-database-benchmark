import re
from .common import InfoExtractor

class EchoMskIE(InfoExtractor):
    _VALID_URL = 'https?://(?:www\\.)?echo\\.msk\\.ru/sounds/(?P<id>\\d+)'
    _TEST = {'url': 'http://www.echo.msk.ru/sounds/1464134.html', 'md5': '2e44b3b78daff5b458e4dbc37f191f7c', 'info_dict': {'id': '1464134', 'ext': 'mp3', 'title': 'Особое мнение - 29 декабря 2014, 19:08'}}

    def _real_extract(self, url):
        if False:
            while True:
                i = 10
        video_id = self._match_id(url)
        webpage = self._download_webpage(url, video_id)
        audio_url = self._search_regex('<a rel="mp3" href="([^"]+)">', webpage, 'audio URL')
        title = self._html_search_regex('<a href="/programs/[^"]+" target="_blank">([^<]+)</a>', webpage, 'title')
        air_date = self._html_search_regex('(?s)<div class="date">(.+?)</div>', webpage, 'date', fatal=False, default=None)
        if air_date:
            air_date = re.sub('(\\s)\\1+', '\\1', air_date)
            if air_date:
                title = '%s - %s' % (title, air_date)
        return {'id': video_id, 'url': audio_url, 'title': title}