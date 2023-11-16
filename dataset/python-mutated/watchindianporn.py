import re
from .common import InfoExtractor
from ..utils import parse_duration

class WatchIndianPornIE(InfoExtractor):
    IE_DESC = 'Watch Indian Porn'
    _VALID_URL = 'https?://(?:www\\.)?watchindianporn\\.net/(?:[^/]+/)*video/(?P<display_id>[^/]+)-(?P<id>[a-zA-Z0-9]+)\\.html'
    _TEST = {'url': 'http://www.watchindianporn.net/video/hot-milf-from-kerala-shows-off-her-gorgeous-large-breasts-on-camera-RZa2avywNPa.html', 'md5': '249589a164dde236ec65832bfce17440', 'info_dict': {'id': 'RZa2avywNPa', 'display_id': 'hot-milf-from-kerala-shows-off-her-gorgeous-large-breasts-on-camera', 'ext': 'mp4', 'title': 'Hot milf from kerala shows off her gorgeous large breasts on camera', 'thumbnail': 're:^https?://.*\\.jpg$', 'duration': 226, 'view_count': int, 'categories': list, 'age_limit': 18}}

    def _real_extract(self, url):
        if False:
            for i in range(10):
                print('nop')
        mobj = self._match_valid_url(url)
        video_id = mobj.group('id')
        display_id = mobj.group('display_id')
        webpage = self._download_webpage(url, display_id)
        info_dict = self._parse_html5_media_entries(url, webpage, video_id)[0]
        title = self._html_search_regex(('<title>(.+?)\\s*-\\s*Indian\\s+Porn</title>', '<h4>(.+?)</h4>'), webpage, 'title')
        duration = parse_duration(self._search_regex('Time:\\s*<strong>\\s*(.+?)\\s*</strong>', webpage, 'duration', fatal=False))
        view_count = int(self._search_regex('(?s)Time:\\s*<strong>.*?</strong>.*?<strong>\\s*(\\d+)\\s*</strong>', webpage, 'view count', fatal=False))
        categories = re.findall('<a[^>]+class=[\\\'"]categories[\\\'"][^>]*>\\s*([^<]+)\\s*</a>', webpage)
        info_dict.update({'id': video_id, 'display_id': display_id, 'http_headers': {'Referer': url}, 'title': title, 'duration': duration, 'view_count': view_count, 'categories': categories, 'age_limit': 18})
        return info_dict