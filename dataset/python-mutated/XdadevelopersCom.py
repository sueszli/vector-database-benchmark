from ..base.simple_downloader import SimpleDownloader

class XdadevelopersCom(SimpleDownloader):
    __name__ = 'XdadevelopersCom'
    __type__ = 'downloader'
    __version__ = '0.08'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?forum\\.xda-developers\\.com/devdb/project/dl/\\?id=\\d+'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Xda-developers.com downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('zapp-brannigan', 'fuerst.reinje@web.de')]
    NAME_PATTERN = '<label>Filename:</label>\\s*<div>\\s*(?P<N>.*?)\\n'
    SIZE_PATTERN = '<label>Size:</label>\\s*<div>\\s*(?P<S>[\\d.,]+)(?P<U>[\\w^_]+)'
    OFFLINE_PATTERN = '</i> Device Filter</h3>'

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
        self.link = pyfile.url + '&task=get'