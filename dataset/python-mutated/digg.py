from .common import InfoExtractor
from ..utils import js_to_json

class DiggIE(InfoExtractor):
    _VALID_URL = 'https?://(?:www\\.)?digg\\.com/video/(?P<id>[^/?#&]+)'
    _TESTS = [{'url': 'http://digg.com/video/sci-fi-short-jonah-daniel-kaluuya-get-out', 'info_dict': {'id': 'LcqvmS0b', 'ext': 'mp4', 'title': "'Get Out' Star Daniel Kaluuya Goes On 'Moby Dick'-Like Journey In Sci-Fi Short 'Jonah'", 'description': 'md5:541bb847648b6ee3d6514bc84b82efda', 'upload_date': '20180109', 'timestamp': 1515530551}, 'params': {'skip_download': True}}, {'url': 'http://digg.com/video/dog-boat-seal-play', 'only_matching': True}, {'url': 'http://digg.com/video/dream-girl-short-film', 'only_matching': True}]

    def _real_extract(self, url):
        if False:
            while True:
                i = 10
        display_id = self._match_id(url)
        webpage = self._download_webpage(url, display_id)
        info = self._parse_json(self._search_regex('(?s)video_info\\s*=\\s*({.+?});\\n', webpage, 'video info', default='{}'), display_id, transform_source=js_to_json, fatal=False)
        video_id = info.get('video_id')
        if video_id:
            provider = info.get('provider_name')
            if provider == 'youtube':
                return self.url_result(video_id, ie='Youtube', video_id=video_id)
            elif provider == 'jwplayer':
                return self.url_result('jwplatform:%s' % video_id, ie='JWPlatform', video_id=video_id)
        return self.url_result(url, 'Generic')