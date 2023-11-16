"""
$description Live TV channels and video on-demand service from RTP, a Portuguese public, state-owned broadcaster.
$url rtp.pt/play
$type live, vod
$region Portugal
"""
import re
from base64 import b64decode
from urllib.parse import unquote
from streamlink.plugin import Plugin, pluginmatcher
from streamlink.plugin.api import useragents, validate
from streamlink.stream.hls import HLSStream

@pluginmatcher(re.compile('https?://www\\.rtp\\.pt/play/'))
class RTPPlay(Plugin):

    def _get_streams(self):
        if False:
            return 10
        self.session.http.headers.update({'User-Agent': useragents.CHROME, 'Referer': self.url})
        re_m3u8 = re.compile('\n                hls\\s*:\\s*(?:\n                    (?P<q>["\'])(?P<string>.*?)(?P=q)\n                    |\n                    decodeURIComponent\\s*\\((?P<obfuscated>\\[.*?])\\.join\\(\n                    |\n                    atob\\s*\\(\\s*decodeURIComponent\\s*\\((?P<obfuscated_b64>\\[.*?])\\.join\\(\n                )\n            ', re.VERBOSE)
        hls_url = self.session.http.get(self.url, schema=validate.Schema(validate.transform(lambda text: next(reversed(list(re_m3u8.finditer(text))), None)), validate.any(None, validate.all(validate.get('string'), str, validate.any('', validate.url())), validate.all(validate.get('obfuscated'), str, validate.parse_json(), validate.transform(lambda arr: unquote(''.join(arr))), validate.url()), validate.all(validate.get('obfuscated_b64'), str, validate.parse_json(), validate.transform(lambda arr: unquote(''.join(arr))), validate.transform(lambda b64: b64decode(b64).decode('utf-8')), validate.url()))))
        if hls_url:
            return HLSStream.parse_variant_playlist(self.session, hls_url)
__plugin__ = RTPPlay