import re
from ..base.simple_downloader import SimpleDownloader

class UptoboxCom(SimpleDownloader):
    __name__ = 'UptoboxCom'
    __type__ = 'downloader'
    __version__ = '0.41'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?(uptobox|uptostream)\\.(?:com|eu|link)/(?P<ID>\\w{12})'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Uptobox.com downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com'), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    PLUGIN_DOMAIN = 'uptobox.link'
    INFO_PATTERN = '(?:"para_title">|<h1(?: .*)?>)(?P<N>.+) \\((?P<S>[\\d.,]+) (?P<U>[\\w^_]+)\\)'
    OFFLINE_PATTERN = '(File not found|Access Denied|404 Not Found)'
    TEMP_OFFLINE_PATTERN = '>Service Unavailable'
    WAIT_PATTERN = 'data-remaining-time=["\'](\\d+)["\']'
    LINK_PATTERN = '[\'"](https?://(?:obwp\\d+\\.uptobox\\.(?:com|eu|link)|\\w+\\.uptobox\\.(?:com|eu|link)/dl?)/.+?)[\'"]'
    DL_LIMIT_PATTERN = 'or you can wait (.+) to launch a new download'
    URL_REPLACEMENTS = [(__pattern__ + '.*', 'https://uptobox.link/\\g<ID>')]

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.multi_dl = self.premium
        self.chunk_limit = 1
        self.resume_download = True

    def handle_free(self, pyfile):
        if False:
            print('Hello World!')
        m = re.search('<input name=["\']waitingToken["\'] value=["\'](.+?)["\']', self.data)
        if m is not None:
            self.data = self.load(pyfile.url, post={'waitingToken': m.group(1), 'submit': 'Free Download'})
        m = re.search(self.LINK_PATTERN, self.data)
        if m is not None:
            self.link = m.group(1)