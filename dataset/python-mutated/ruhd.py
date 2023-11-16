from .common import InfoExtractor

class RUHDIE(InfoExtractor):
    _VALID_URL = 'https?://(?:www\\.)?ruhd\\.ru/play\\.php\\?vid=(?P<id>\\d+)'
    _TEST = {'url': 'http://www.ruhd.ru/play.php?vid=207', 'md5': 'd1a9ec4edf8598e3fbd92bb16072ba83', 'info_dict': {'id': '207', 'ext': 'divx', 'title': 'КОТ бааааам', 'description': 'классный кот)', 'thumbnail': 're:^http://.*\\.jpg$'}}

    def _real_extract(self, url):
        if False:
            return 10
        video_id = self._match_id(url)
        webpage = self._download_webpage(url, video_id)
        video_url = self._html_search_regex('<param name="src" value="([^"]+)"', webpage, 'video url')
        title = self._html_search_regex('<title>([^<]+)&nbsp;&nbsp; RUHD\\.ru - Видео Высокого качества №1 в России!</title>', webpage, 'title')
        description = self._html_search_regex('(?s)<div id="longdesc">(.+?)<span id="showlink">', webpage, 'description', fatal=False)
        thumbnail = self._html_search_regex('<param name="previewImage" value="([^"]+)"', webpage, 'thumbnail', fatal=False)
        if thumbnail:
            thumbnail = 'http://www.ruhd.ru' + thumbnail
        return {'id': video_id, 'url': video_url, 'title': title, 'description': description, 'thumbnail': thumbnail}