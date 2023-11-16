from ..base.xfs_downloader import XFSDownloader

class DropDownload(XFSDownloader):
    __name__ = 'DropDownload'
    __type__ = 'downloader'
    __version__ = '0.03'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?drop\\.download/\\w{12}'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Drop.download downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    PLUGIN_DOMAIN = 'drop.download'
    LINK_PATTERN = '<a href="(https://s\\d+\\.drop\\.download.+?)"'

    def setup(self):
        if False:
            return 10
        self.multi_dl = True
        self.resume_download = True
        self.chunk_limit = -1