import json
import re
from pyload.core.network.http.http_request import HTTPRequest
from ..base.simple_downloader import SimpleDownloader

class SoundcloudCom(SimpleDownloader):
    __name__ = 'SoundcloudCom'
    __type__ = 'downloader'
    __version__ = '0.22'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?soundcloud\\.com/[\\w\\-]+/[\\w\\-]+'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'SoundCloud.com downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com'), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    NAME_PATTERN = 'title" content="(?P<N>.+?)"'
    OFFLINE_PATTERN = '<title>"SoundCloud - Hear the world’s sounds"</title>'

    def setup(self):
        if False:
            print('Hello World!')
        try:
            self.req.http.close()
        except Exception:
            pass
        self.req.http = HTTPRequest(cookies=self.req.cj, options=self.pyload.request_factory.get_options(), limit=5000000)

    def get_info(self, url='', html=''):
        if False:
            for i in range(10):
                print('nop')
        info = super(SoundcloudCom, self).get_info(url, html)
        if 'name' in info:
            info['name'] += '.mp3'
        return info

    def handle_free(self, pyfile):
        if False:
            i = 10
            return i + 15
        try:
            json_data = re.search('<script>window\\.__sc_hydration = (.+?);</script>', self.data).group(1)
        except (AttributeError, IndexError):
            self.fail('Failed to retrieve json_data')
        try:
            js_url = re.findall('script crossorigin src="(.+?)"></script>', self.data)[-1]
        except IndexError:
            self.fail(self._('Failed to find js_url'))
        js_data = self.load(js_url)
        m = re.search('[ ,]client_id:"(.+?)"', js_data)
        if m is None:
            self.fail(self._('client_id not found'))
        client_id = m.group(1)
        hydra_table = {table['hydratable']: table['data'] for table in json.loads(json_data)}
        streams = [s['url'] for s in hydra_table['sound']['media']['transcodings'] if s['format']['protocol'] == 'progressive' and s['format']['mime_type'] == 'audio/mpeg']
        track_authorization = hydra_table['sound']['track_authorization']
        if streams:
            json_data = self.load(streams[0], get={'client_id': client_id, 'track_authorization': track_authorization})
            self.link = json.loads(json_data).get('url')