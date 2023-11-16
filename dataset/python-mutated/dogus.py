"""
$description Turkish live TV channels from Dogus Group, including Euro Star, Star and NTV.
$url eurostartv.com.tr
$url kralmuzik.com.tr
$url ntv.com.tr
$url startv.com.tr
$type live
"""
import re
from streamlink.plugin import Plugin, pluginmatcher
from streamlink.plugin.api import validate
from streamlink.stream.hls import HLSStream

@pluginmatcher(re.compile('https?://(?:www\\.)?eurostartv\\.com\\.tr/canli-izle'))
@pluginmatcher(re.compile('https?://(?:www\\.)?kralmuzik\\.com\\.tr/tv/.+'))
@pluginmatcher(re.compile('https?://(?:www\\.)?ntv\\.com\\.tr/canli-yayin/ntv'))
@pluginmatcher(re.compile('https?://(?:www\\.)?startv\\.com\\.tr/canli-yayin'))
class Dogus(Plugin):
    _re_live_hls = re.compile("'(https?://[^']+/live/hls/[^']+)'")
    _re_yt_script = re.compile("youtube\\.init\\('([\\w-]{11})'")

    def _get_streams(self):
        if False:
            print('Hello World!')
        root = self.session.http.get(self.url, schema=validate.Schema(validate.parse_html()))
        yt_iframe = root.xpath("string(.//iframe[contains(@src,'youtube.com')][1]/@src)")
        dm_iframe = root.xpath("string(.//iframe[contains(@src,'dailymotion.com')][1]/@src)")
        yt_script = root.xpath("string(.//script[contains(text(), 'youtube.init')][1]/text())")
        if yt_script:
            m = self._re_yt_script.search(yt_script)
            if m:
                yt_iframe = f'https://www.youtube.com/watch?v={m.group(1)}'
        iframe = yt_iframe or dm_iframe
        if iframe:
            return self.session.streams(iframe)
        dd_script = root.xpath("string(.//script[contains(text(), '/live/hls/')][1]/text())")
        if dd_script:
            m = self._re_live_hls.search(dd_script)
            if m:
                return HLSStream.parse_variant_playlist(self.session, m.group(1))
__plugin__ = Dogus