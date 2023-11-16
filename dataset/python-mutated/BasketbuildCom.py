import re
from ..base.simple_downloader import SimpleDownloader

class BasketbuildCom(SimpleDownloader):
    __name__ = 'BasketbuildCom'
    __type__ = 'downloader'
    __version__ = '0.08'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?(?:\\w\\.)?basketbuild\\.com/filedl/.+'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Basketbuild.com downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('zapp-brannigan', 'fuerst.reinje@web.de')]
    NAME_PATTERN = 'File Name:</strong> (?P<N>.+?)<br/>'
    SIZE_PATTERN = 'File Size:</strong> (?P<S>[\\d.,]+) (?P<U>[\\w^_]+)'
    OFFLINE_PATTERN = '404 - Page Not Found'

    def setup(self):
        if False:
            print('Hello World!')
        self.multi_dl = True
        self.resume_download = True
        self.chunk_limit = 1

    def handle_free(self, pyfile):
        if False:
            i = 10
            return i + 15
        try:
            link1 = re.search('href="(.+dlgate/.+)"', self.data).group(1)
            self.data = self.load(link1)
        except AttributeError:
            self.error(self._('Hop #1 not found'))
        else:
            self.log_debug(f'Next hop: {link1}')
        try:
            wait = re.search('var sec = (\\d+)', self.data).group(1)
            self.log_debug(f'Wait {wait} seconds')
            self.wait(wait)
        except AttributeError:
            self.log_debug('No wait time found')
        try:
            self.link = re.search('id="dlLink">\\s*<a href="(.+?)"', self.data).group(1)
        except AttributeError:
            self.error(self._('DL-Link not found'))