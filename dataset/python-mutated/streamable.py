"""
$description Global video hosting platform.
$url streamable.com
$type vod
"""
import re
from streamlink.plugin import Plugin, pluginmatcher
from streamlink.plugin.api import validate
from streamlink.stream.http import HTTPStream
from streamlink.utils.url import update_scheme

@pluginmatcher(re.compile('https?://(?:www\\.)?streamable\\.com/(.+)'))
class Streamable(Plugin):

    def _get_streams(self):
        if False:
            while True:
                i = 10
        data = self.session.http.get(self.url, schema=validate.Schema(re.compile('var\\s*videoObject\\s*=\\s*({.*?});'), validate.none_or_all(validate.get(1), validate.parse_json(), {'files': {str: {'url': validate.url(), 'width': int, 'height': int, 'bitrate': int}}})))
        for info in data['files'].values():
            stream_url = update_scheme('https://', info['url'])
            res = min(info['width'], info['height'])
            yield (f'{res}p', HTTPStream(self.session, stream_url))
__plugin__ = Streamable