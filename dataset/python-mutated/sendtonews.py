import re
from .common import InfoExtractor
from ..utils import float_or_none, parse_iso8601, update_url_query, int_or_none, determine_protocol, unescapeHTML

class SendtoNewsIE(InfoExtractor):
    _VALID_URL = 'https?://embed\\.sendtonews\\.com/player2/embedplayer\\.php\\?.*\\bSC=(?P<id>[0-9A-Za-z-]+)'
    _TEST = {'url': 'http://embed.sendtonews.com/player2/embedplayer.php?SC=GxfCe0Zo7D-175909-5588&type=single&autoplay=on&sound=YES', 'info_dict': {'id': 'GxfCe0Zo7D-175909-5588'}, 'playlist_count': 8, 'playlist': [{'info_dict': {'id': '240385', 'ext': 'mp4', 'title': 'Indians introduce Encarnacion', 'description': "Indians president of baseball operations Chris Antonetti and Edwin Encarnacion discuss the slugger's three-year contract with Cleveland", 'duration': 137.898, 'thumbnail': 're:https?://.*\\.jpg$', 'upload_date': '20170105', 'timestamp': 1483649762}}], 'params': {'skip_download': True}}
    _URL_TEMPLATE = '//embed.sendtonews.com/player2/embedplayer.php?SC=%s'

    @classmethod
    def _extract_embed_urls(cls, url, webpage):
        if False:
            print('Hello World!')
        mobj = re.search('(?x)<script[^>]+src=([\\\'"])\n            (?:https?:)?//embed\\.sendtonews\\.com/player/responsiveembed\\.php\\?\n                .*\\bSC=(?P<SC>[0-9a-zA-Z-]+).*\n            \\1>', webpage)
        if mobj:
            sc = mobj.group('SC')
            yield (cls._URL_TEMPLATE % sc)

    def _real_extract(self, url):
        if False:
            i = 10
            return i + 15
        playlist_id = self._match_id(url)
        data_url = update_url_query(url.replace('embedplayer.php', 'data_read.php'), {'cmd': 'loadInitial'})
        playlist_data = self._download_json(data_url, playlist_id)
        entries = []
        for video in playlist_data['playlistData'][0]:
            info_dict = self._parse_jwplayer_data(video['jwconfiguration'], require_title=False, m3u8_id='hls', rtmp_params={'no_resume': True})
            for f in info_dict['formats']:
                if f.get('tbr'):
                    continue
                tbr = int_or_none(self._search_regex('/(\\d+)k/', f['url'], 'bitrate', default=None))
                if not tbr:
                    continue
                f.update({'format_id': '%s-%d' % (determine_protocol(f), tbr), 'tbr': tbr})
            thumbnails = []
            if video.get('thumbnailUrl'):
                thumbnails.append({'id': 'normal', 'url': video['thumbnailUrl']})
            if video.get('smThumbnailUrl'):
                thumbnails.append({'id': 'small', 'url': video['smThumbnailUrl']})
            info_dict.update({'title': video['S_headLine'].strip(), 'description': unescapeHTML(video.get('S_fullStory')), 'thumbnails': thumbnails, 'duration': float_or_none(video.get('SM_length')), 'timestamp': parse_iso8601(video.get('S_sysDate'), delimiter=' '), '_format_sort_fields': ('tbr', 'res')})
            entries.append(info_dict)
        return self.playlist_result(entries, playlist_id)