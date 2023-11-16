from .common import InfoExtractor

class MyChannelsIE(InfoExtractor):
    _VALID_URL = 'https?://(?:www\\.)?mychannels\\.com/.*(?P<id_type>video|production)_id=(?P<id>[0-9]+)'
    _TEST = {'url': 'https://mychannels.com/missholland/miss-holland?production_id=3416', 'md5': 'b8993daad4262dd68d89d651c0c52c45', 'info_dict': {'id': 'wUUDZZep6vQD', 'ext': 'mp4', 'title': 'Miss Holland joins VOTE LEAVE', 'description': 'Miss Holland | #13 Not a potato', 'uploader': 'Miss Holland'}}

    def _real_extract(self, url):
        if False:
            for i in range(10):
                print('nop')
        (id_type, url_id) = self._match_valid_url(url).groups()
        webpage = self._download_webpage(url, url_id)
        video_data = self._html_search_regex('<div([^>]+data-%s-id="%s"[^>]+)>' % (id_type, url_id), webpage, 'video data')

        def extract_data_val(attr, fatal=False):
            if False:
                print('Hello World!')
            return self._html_search_regex('data-%s\\s*=\\s*"([^"]+)"' % attr, video_data, attr, fatal=fatal)
        minoto_id = extract_data_val('minoto-id') or self._search_regex('/id/([a-zA-Z0-9]+)', extract_data_val('video-src', True), 'minoto id')
        return {'_type': 'url_transparent', 'url': 'minoto:%s' % minoto_id, 'id': url_id, 'title': extract_data_val('title', True), 'description': extract_data_val('description'), 'thumbnail': extract_data_val('image'), 'uploader': extract_data_val('channel')}