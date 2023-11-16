from ..base.simple_downloader import SimpleDownloader

class SenditCloud(SimpleDownloader):
    __name__ = 'SenditCloud'
    __type__ = 'downloader'
    __version__ = '0.01'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?sendit\\.cloud/\\w+'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Sendit.cloud downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    NAME_PATTERN = '<header class="widget-header ">.+>\\s*(.+?)\\s*</header>'
    SIZE_PATTERN = '<b>Download</b> <font color="#FFFFFF">\\((?P<S>[\\d.,]+) (?P<U>[\\w_^]+)\\)<'
    OFFLINE_PATTERN = 'The file you are trying to download is no longer available'

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.multi_dl = True
        self.resume_download = False

    def handle_free(self, pyfile):
        if False:
            while True:
                i = 10
        (url, inputs) = self.parse_html_form('name="F1"')
        if inputs is not None:
            self.download(pyfile.url, post=inputs)