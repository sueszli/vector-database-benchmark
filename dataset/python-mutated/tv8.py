"""
$description Turkish live TV channel owned by Acun Medya Group.
$url tv8.com.tr
$type live
"""
import logging
import re
from streamlink.plugin import Plugin, pluginmatcher
from streamlink.plugin.api import validate
from streamlink.stream.hls import HLSStream, HLSStreamReader, HLSStreamWriter
log = logging.getLogger(__name__)

class TV8HLSStreamWriter(HLSStreamWriter):
    ad_re = re.compile('/ad/|/crea/')

    def should_filter_segment(self, segment):
        if False:
            print('Hello World!')
        return self.ad_re.search(segment.uri) is not None or super().should_filter_segment(segment)

class TV8HLSStreamReader(HLSStreamReader):
    __writer__ = TV8HLSStreamWriter

class TV8HLSStream(HLSStream):
    __reader__ = TV8HLSStreamReader

@pluginmatcher(re.compile('https?://www\\.tv8\\.com\\.tr/canli-yayin'))
class TV8(Plugin):

    def _get_streams(self):
        if False:
            while True:
                i = 10
        hls_url = self.session.http.get(self.url, schema=validate.Schema(re.compile('var\\s+videoUrl\\s*=\\s*(?P<q>["\'])(?P<hls_url>https?://.*?\\.m3u8.*?)(?P=q)'), validate.any(None, validate.get('hls_url'))))
        if hls_url is not None:
            return TV8HLSStream.parse_variant_playlist(self.session, hls_url)
__plugin__ = TV8