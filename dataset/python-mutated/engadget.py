from .common import InfoExtractor

class EngadgetIE(InfoExtractor):
    _VALID_URL = 'https?://(?:www\\.)?engadget\\.com/video/(?P<id>[^/?#]+)'
    _TESTS = [{'url': 'https://www.engadget.com/video/57a28462134aa15a39f0421a/', 'only_matching': True}]

    def _real_extract(self, url):
        if False:
            print('Hello World!')
        video_id = self._match_id(url)
        return self.url_result('aol-video:%s' % video_id)