from ..base.simple_downloader import SimpleDownloader
import re

class SolidfilesCom(SimpleDownloader):
    __name__ = 'SolidfilesCom'
    __type__ = 'downloader'
    __version__ = '0.09'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?solidfiles\\.com\\/[dv]/\\w+'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Solidfiles.com downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('sraedler', 'simon.raedler@yahoo.de'), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    NAME_PATTERN = '<h1 class="node-name">(?P<N>.+?)</h1>'
    SIZE_PATTERN = '</copy-button>\\s*(?P<S>[\\d.,]+) (?P<U>[\\w_^]+)'
    OFFLINE_PATTERN = '<h1>404'

    def setup(self):
        if False:
            print('Hello World!')
        self.multi_dl = True
        self.chunk_limit = 1

    def handle_free(self, pyfile):
        if False:
            i = 10
            return i + 15
        m = re.search('"downloadUrl":"(.+?)"', self.data)
        if m is not None:
            self.link = m.group(1)