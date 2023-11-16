"""
$description Live TV channels and video on-demand service from RUV, an Icelandic public, state-owned broadcaster.
$url ruv.is
$type live, vod
$region Iceland
"""
import re
from streamlink.plugin import Plugin, pluginmatcher
from streamlink.stream.hls import HLSStream
RUV_LIVE_API = 'http://www.ruv.is/sites/all/themes/at_ruv/scripts/ruv-stream.php?channel={0}&format=json'
_single_re = re.compile('(?P<url>http://[0-9a-zA-Z\\-\\.]*/\n                            (lokad|opid)\n                            /\n                            ([0-9]+/[0-9][0-9]/[0-9][0-9]/)?\n                            ([A-Z0-9\\$_]+\\.mp4\\.m3u8)\n                            )\n                         ', re.VERBOSE)
_multi_re = re.compile('(?P<base_url>http://[0-9a-zA-Z\\-\\.]*/\n                            (lokad|opid)\n                            /)\n                            manifest.m3u8\\?tlm=hls&streams=\n                            (?P<streams>[0-9a-zA-Z\\/\\.\\,:]+)\n                         ', re.VERBOSE)

@pluginmatcher(re.compile('\n    https?://(?:www\\.)?ruv\\.is/\n    (?P<stream_id>ruv|ruv2|ruv-2|ras1|ras2|rondo)\n    /?$\n', re.VERBOSE))
@pluginmatcher(re.compile('\n    https?://(?:www\\.)?ruv\\.is/spila/\n    (?P<stream_id>ruv|ruv2|ruv-2|ruv-aukaras)\n    /[a-zA-Z0-9_-]+\n    /[0-9]+\n    /?\n', re.VERBOSE))
class Ruv(Plugin):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.live = self.matches[0] is not None
        if self.live:
            self.stream_id = self.match.group('stream_id').replace('-', '')
            if self.stream_id == 'rondo':
                self.stream_id = 'ras3'

    def _get_live_streams(self):
        if False:
            while True:
                i = 10
        res = self.session.http.get(RUV_LIVE_API.format(self.stream_id))
        json_res = self.session.http.json(res)
        for url in json_res['result']:
            if url.startswith('rtmp:'):
                continue
            streams = HLSStream.parse_variant_playlist(self.session, url)
            yield from streams.items()

    def _get_sarpurinn_streams(self):
        if False:
            while True:
                i = 10
        res = self.session.http.get(self.url).text
        lines = '\n'.join([line for line in res.split('\n') if 'video.src' in line])
        multi_stream_match = _multi_re.search(lines)
        if multi_stream_match and multi_stream_match.group('streams'):
            base_url = multi_stream_match.group('base_url')
            streams = multi_stream_match.group('streams').split(',')
            for stream in streams:
                if stream.count(':') != 1:
                    continue
                [token, quality] = stream.split(':')
                quality = int(quality)
                key = ''
                if quality <= 500:
                    key = '240p'
                elif quality <= 800:
                    key = '360p'
                elif quality <= 1200:
                    key = '480p'
                elif quality <= 2400:
                    key = '720p'
                else:
                    key = '1080p'
                yield (key, HLSStream(self.session, base_url + token))
        else:
            single_stream_match = _single_re.search(lines)
            if single_stream_match:
                url = single_stream_match.group('url')
                yield ('576p', HLSStream(self.session, url))

    def _get_streams(self):
        if False:
            return 10
        if self.live:
            return self._get_live_streams()
        else:
            return self._get_sarpurinn_streams()
__plugin__ = Ruv