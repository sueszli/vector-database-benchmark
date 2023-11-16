"""
$description German sports magazine live stream, owned by ARD.
$url sportschau.de
$type live
"""
import logging
import re
from streamlink.plugin import Plugin, pluginmatcher
from streamlink.plugin.api import validate
from streamlink.stream.hls import HLSStream
from streamlink.stream.http import HTTPStream
from streamlink.utils.url import update_scheme
log = logging.getLogger(__name__)

@pluginmatcher(re.compile('https?://(?:\\w+\\.)*sportschau\\.de/'))
class Sportschau(Plugin):

    def _get_streams(self):
        if False:
            for i in range(10):
                print('nop')
        player_js = self.session.http.get(self.url, schema=validate.Schema(re.compile('https?:(//deviceids-medp.wdr.de/ondemand/\\S+\\.js)'), validate.none_or_all(validate.get(1), validate.transform(lambda url: update_scheme('https://', url)))))
        if not player_js:
            return
        log.debug(f'Found player js {player_js}')
        data = self.session.http.get(player_js, schema=validate.Schema(validate.regex(re.compile('\\$mediaObject\\.jsonpHelper\\.storeAndPlay\\(({.+})\\);?')), validate.get(1), validate.parse_json(), validate.get('mediaResource'), validate.get('dflt'), {validate.optional('audioURL'): validate.url(), validate.optional('videoURL'): validate.url()}))
        if data.get('videoURL'):
            yield from HLSStream.parse_variant_playlist(self.session, update_scheme('https:', data.get('videoURL'))).items()
        if data.get('audioURL'):
            yield ('audio', HTTPStream(self.session, update_scheme('https:', data.get('audioURL'))))
__plugin__ = Sportschau