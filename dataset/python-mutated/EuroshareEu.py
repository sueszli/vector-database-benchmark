import re
from datetime import timedelta
from ..base.simple_downloader import SimpleDownloader

class EuroshareEu(SimpleDownloader):
    __name__ = 'EuroshareEu'
    __type__ = 'downloader'
    __version__ = '0.41'
    __status__ = 'testing'
    __pattern__ = 'http://(?:www\\.)?euroshare\\.(eu|sk|cz|hu|pl)/file/.+'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Euroshare.eu downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('zoidberg', 'zoidberg@mujmail.cz'), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    NAME_PATTERN = '<div class="main-panel__title">(?P<N>.+?)</div>'
    SIZE_PATTERN = 'File size: (?P<S>[\\d.,]+) (?P<U>[\\w^_]+)'
    OFFLINE_PATTERN = '<h2>S.bor sa nena.iel</h2>|<div class="container">FILE NOT FOUND</div>|>&nbsp;File has been removed due to inactivity\\.<'
    TEMP_OFFLINE_PATTERN = '^unmatchable$'
    LINK_PATTERN = '<a href="(http://\\w+\\.euroshare\\.eu/file/.+?\\?download_token=\\w+)"'
    DL_LIMIT_PATTERN = '<h2>Prebieha s.ahovanie</h2>|<p>Naraz je z jednej IP adresy mo.n. s.ahova. iba jeden s.bor'
    ERROR_PATTERN = 'href="/customer-zone/login/"'
    URL_REPLACEMENTS = [('(http://[^/]*\\.)(sk|cz|hu|pl)/', '\\1eu/')]

    def setup(self):
        if False:
            while True:
                i = 10
        self.resume_download = True

    def handle_free(self, pyfile):
        if False:
            return 10
        if re.search(self.DL_LIMIT_PATTERN, self.data):
            self.retry(timedelta(minutes=5).total_seconds(), 12, self._('Download limit reached'))
        self.data = self.load(pyfile.url, get={'download': 'true'})
        m = re.search(self.LINK_PATTERN, self.data)
        if m is None:
            self.log_debug(self.data)
            self.error(self._('LINK_PATTERN not found'))
        self.link = m.group(1)