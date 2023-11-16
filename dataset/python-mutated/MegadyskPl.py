import base64
import json
import re
import urllib.parse
from ..base.simple_downloader import SimpleDownloader

def xor_decrypt(data, key):
    if False:
        for i in range(10):
            print('nop')
    data = base64.b64decode(data)
    return ''.join([chr(ord(x[1]) ^ ord(key[x[0].format(len(key))])) for x in [(i, c) for (i, c) in enumerate(data)]])

class MegadyskPl(SimpleDownloader):
    __name__ = 'MegadyskPl'
    __type__ = 'downloader'
    __version__ = '0.05'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?megadysk\\.pl/dl/.+'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Megadysk.pl downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    NAME_PATTERN = 'data-reactid="25">(?P<N>.+?)<'
    SIZE_PATTERN = '<!-- react-text: 40 -->(?P<S>[\\d.,]+)(?P<U>[\\w^_]+)'
    OFFLINE_PATTERN = '(?:Nothing has been found|have been deleted)<'

    def api_info(self, url):
        if False:
            print('Hello World!')
        html = self.load(url)
        info = {}
        m = re.search('window\\[\'.*?\'\\]\\s*=\\s*\\"(.*?)\\"', html)
        if m is None:
            info['status'] = 8
            info['error'] = 'Encrypted info pattern not found'
            return info
        encrypted_info = m.group(1)
        html = self.load('https://megadysk.pl/dist/index.js')
        m = re.search('t.ISK\\s*=\\s*"(\\w+)"', html)
        if m is None:
            info['status'] = 8
            info['error'] = 'Encryption key pattern not found'
            return info
        key = m.group(1)
        res = xor_decrypt(encrypted_info, key)
        json_data = json.loads(urllib.parse.unquote(res))
        if json_data['app']['maintenance']:
            info['status'] = 6
            return info
        if json_data['app']['downloader'] is None or json_data['app']['downloader']['file']['deleted']:
            info['status'] = 1
            return info
        info['name'] = json_data['app']['downloader']['file']['name']
        info['size'] = json_data['app']['downloader']['file']['size']
        info['download_url'] = json_data['app']['downloader']['url']
        return info

    def setup(self):
        if False:
            while True:
                i = 10
        self.multi_dl = True
        self.resume_download = False
        self.chunk_limit = 1

    def handle_free(self, pyfile):
        if False:
            i = 10
            return i + 15
        if 'download_url' not in self.info:
            self.error(self._('Missing JSON data'))
        self.link = self.fixurl(self.info['download_url'])