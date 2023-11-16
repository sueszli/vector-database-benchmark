from .common import InfoExtractor
from ..utils import unescapeHTML

class VODPlatformIE(InfoExtractor):
    _VALID_URL = 'https?://(?:(?:www\\.)?vod-platform\\.net|embed\\.kwikmotion\\.com)/[eE]mbed/(?P<id>[^/?#]+)'
    _EMBED_REGEX = ['<iframe[^>]+src=(["\\\'])(?P<url>(?:https?:)?//(?:(?:www\\.)?vod-platform\\.net|embed\\.kwikmotion\\.com)/[eE]mbed/.+?)\\1']
    _TESTS = [{'url': 'http://vod-platform.net/embed/RufMcytHDolTH1MuKHY9Fw', 'md5': '1db2b7249ce383d6be96499006e951fc', 'info_dict': {'id': 'RufMcytHDolTH1MuKHY9Fw', 'ext': 'mp4', 'title': 'LBCi News_ النصرة في ضيافة الـ "سي.أن.أن"'}}, {'url': 'http://embed.kwikmotion.com/embed/RufMcytHDolTH1MuKHY9Fw', 'only_matching': True}]

    def _real_extract(self, url):
        if False:
            return 10
        video_id = self._match_id(url)
        webpage = self._download_webpage(url, video_id)
        title = unescapeHTML(self._og_search_title(webpage))
        hidden_inputs = self._hidden_inputs(webpage)
        formats = self._extract_wowza_formats(hidden_inputs.get('HiddenmyhHlsLink') or hidden_inputs['HiddenmyDashLink'], video_id, skip_protocols=['f4m', 'smil'])
        return {'id': video_id, 'title': title, 'thumbnail': hidden_inputs.get('HiddenThumbnail') or self._og_search_thumbnail(webpage), 'formats': formats}