"""
$description French live TV channels from TF1 Group, including LCI and TF1.
$url tf1.fr
$url tf1info.fr
$url lci.fr
$type live
$region France
"""
import logging
import re
from streamlink.plugin import Plugin, PluginError, pluginmatcher
from streamlink.plugin.api import useragents, validate
from streamlink.stream.hls import HLSStream
log = logging.getLogger(__name__)

@pluginmatcher(re.compile('\n    https?://(?:www\\.)?\n    (?:\n        tf1\\.fr/(?:\n            (?P<live>[\\w-]+)/direct/?\n            |\n            stream/(?P<stream>[\\w-]+)\n        )\n        |\n        (?P<lci>tf1info|lci)\\.fr/direct/?\n    )\n', re.VERBOSE))
class TF1(Plugin):
    _URL_API = 'https://mediainfo.tf1.fr/mediainfocombo/{channel_id}'

    def _get_channel(self):
        if False:
            i = 10
            return i + 15
        if self.match['live']:
            channel = self.match['live']
            channel_id = f'L_{channel.upper()}'
        elif self.match['lci']:
            channel = 'LCI'
            channel_id = 'L_LCI'
        elif self.match['stream']:
            channel = self.match['stream']
            channel_id = f'L_FAST_v2l-{channel}'
        else:
            raise PluginError('Invalid channel')
        return (channel, channel_id)

    def _api_call(self, channel_id):
        if False:
            print('Hello World!')
        return self.session.http.get(self._URL_API.format(channel_id=channel_id), params={'context': 'MYTF1', 'pver': '4001000'}, headers={'User-Agent': useragents.IPHONE}, schema=validate.Schema(validate.parse_json(), {'delivery': validate.any(validate.all({'code': 200, 'format': 'hls', 'url': validate.url()}, validate.union_get('code', 'url')), validate.all({'code': int, 'error': str}, validate.union_get('code', 'error')))}, validate.get('delivery')))

    def _get_streams(self):
        if False:
            return 10
        (channel, channel_id) = self._get_channel()
        log.debug(f'Found channel {channel} ({channel_id})')
        (code, data) = self._api_call(channel_id)
        if code != 200:
            log.error(data)
            return
        return HLSStream.parse_variant_playlist(self.session, data)
__plugin__ = TF1