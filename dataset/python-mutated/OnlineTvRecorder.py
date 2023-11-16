from pyload.core.network.http.exceptions import BadHeader
from .Http import Http

class OnlineTvRecorder(Http):
    __name__ = 'OnlineTvRecorder'
    __type__ = 'downloader'
    __version__ = '0.05'
    __status__ = 'testing'
    __pattern__ = 'https?://(81\\.95\\.11\\.\\d{1,2}|93\\.115\\.84\\.162|download\\d{1,2}.onlinetvrecorder.com)/download/\\d+/\\d+/\\d*/[0-9a-f]+/.+'
    __config__ = [('enabled', 'bool', 'Activated', True)]
    __description__ = 'OnlineTvRecorder downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Tim Gregory', 'bogeyman@valar.de')]

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.multi_dl = False
        self.chunk_limit = 1
        self.resume_download = True

    def process(self, pyfile):
        if False:
            for i in range(10):
                print('nop')
        try:
            return super().process(pyfile)
        except BadHeader as exc:
            self.log_debug(f'OnlineTvRecorder httpcode: {exc.code}')
            if exc.code == 503:
                self.retry(360, 30, self._('Waiting in download queue'))