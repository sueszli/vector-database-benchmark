from ..base.simple_downloader import SimpleDownloader

class AnonfilesCom(SimpleDownloader):
    __name__ = 'AnonfilesCom'
    __type__ = 'downloader'
    __version__ = '0.02'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?anonfiles?\\.com/(?P<ID>\\w+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Anonfiles.com downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    NAME_PATTERN = 'class="text-center text-wordwrap">(?P<N>.+?)<'
    SIZE_PATTERN = 'Download\\s*\\((?P<S>[\\d.,]+) (?P<U>[\\w^_]+)\\)'
    LINK_PATTERN = 'href="(https://cdn-\\d+.anonfiles.com/.+?)"'
    URL_REPLACEMENTS = [(__pattern__ + '.*', 'https://anonfiles.com/\\g<ID>')]

    def setup(self):
        if False:
            return 10
        self.multi_dl = True
        self.resume_download = True
        self.chunk_limit = -1