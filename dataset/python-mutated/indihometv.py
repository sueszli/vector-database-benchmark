"""
$description Live TV channels and video on-demand service from IndiHome TV, owned by Telkom Indonesia.
$url indihometv.com
$type live, vod
$region Indonesia
"""
import re
from streamlink.plugin import Plugin, pluginmatcher
from streamlink.plugin.api import validate
from streamlink.stream.dash import DASHStream
from streamlink.stream.hls import HLSStream

@pluginmatcher(re.compile('https?://(?:www\\.)?indihometv\\.com/'))
class IndiHomeTV(Plugin):

    def _get_streams(self):
        if False:
            i = 10
            return i + 15
        url = self.session.http.get(self.url, schema=validate.Schema(validate.parse_html(), validate.any(validate.all(validate.xml_xpath_string("\n                        .//script[contains(text(), 'laylist.m3u8') or contains(text(), 'manifest.mpd')][1]/text()\n                    "), str, re.compile('(?P<q>[\'"])(?P<url>https://.*?/(?:[Pp]laylist\\.m3u8|manifest\\.mpd).+?)(?P=q)'), validate.none_or_all(validate.get('url'), validate.url())), validate.all(validate.xml_xpath_string(".//video[@id='video-player'][1]/source[1]/@src"), validate.none_or_all(validate.url())))))
        if url and '.m3u8' in url:
            return HLSStream.parse_variant_playlist(self.session, url)
        elif url and '.mpd' in url:
            return DASHStream.parse_manifest(self.session, url)
__plugin__ = IndiHomeTV