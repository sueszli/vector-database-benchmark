"""
$description Current affairs and cultural channel owned by NHK, a Japanese public, state-owned broadcaster.
$url nhk.or.jp/nhkworld
$type live
"""
import re
from streamlink.plugin import Plugin, pluginmatcher
from streamlink.stream.hls import HLSStream
API_URL = 'http://{}.nhk.or.jp/nhkworld/app/tv/hlslive_web.json'

@pluginmatcher(re.compile('https?://(?:(\\w+)\\.)?nhk\\.or\\.jp/nhkworld'))
class NHKWorld(Plugin):

    def _get_streams(self):
        if False:
            while True:
                i = 10
        sdomain = self.match.group(1) or 'www'
        res = self.session.http.get(API_URL.format(sdomain))
        stream_url = self.session.http.json(res)['main']['wstrm']
        return HLSStream.parse_variant_playlist(self.session, stream_url)
__plugin__ = NHKWorld