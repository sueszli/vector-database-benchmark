from .common import InfoExtractor

class OktoberfestTVIE(InfoExtractor):
    _VALID_URL = 'https?://(?:www\\.)?oktoberfest-tv\\.de/[^/]+/[^/]+/video/(?P<id>[^/?#]+)'
    _TEST = {'url': 'http://www.oktoberfest-tv.de/de/kameras/video/hb-zelt', 'info_dict': {'id': 'hb-zelt', 'ext': 'mp4', 'title': 're:^Live-Kamera: Hofbräuzelt [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}$', 'thumbnail': 're:^https?://.*\\.jpg$', 'is_live': True}, 'params': {'skip_download': True}}

    def _real_extract(self, url):
        if False:
            i = 10
            return i + 15
        video_id = self._match_id(url)
        webpage = self._download_webpage(url, video_id)
        title = self._html_search_regex('<h1><strong>.*?</strong>(.*?)</h1>', webpage, 'title')
        clip = self._search_regex("clip:\\s*\\{\\s*url:\\s*'([^']+)'", webpage, 'clip')
        ncurl = self._search_regex("netConnectionUrl:\\s*'([^']+)'", webpage, 'rtmp base')
        video_url = ncurl + clip
        thumbnail = self._search_regex("canvas:\\s*\\{\\s*backgroundImage:\\s*'url\\(([^)]+)\\)'", webpage, 'thumbnail', fatal=False)
        return {'id': video_id, 'title': title, 'url': video_url, 'ext': 'mp4', 'is_live': True, 'thumbnail': thumbnail}