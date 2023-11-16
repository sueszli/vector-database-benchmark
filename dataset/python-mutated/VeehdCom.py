import re
from ..base.downloader import BaseDownloader

class VeehdCom(BaseDownloader):
    __name__ = 'VeehdCom'
    __type__ = 'downloader'
    __version__ = '0.29'
    __status__ = 'testing'
    __pattern__ = 'http://veehd\\.com/video/\\d+_\\S+'
    __config__ = [('enabled', 'bool', 'Activated', True), ('filename_spaces', 'bool', 'Allow spaces in filename', False), ('replacement_char', 'str', 'Filename replacement character', '_')]
    __description__ = 'Veehd.com downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('cat', 'cat@pyload')]

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.multi_dl = True
        self.req.can_continue = True

    def process(self, pyfile):
        if False:
            print('Hello World!')
        self.download_html()
        if not self.file_exists():
            self.offline()
        pyfile.name = self.get_file_name()
        self.download(self.get_file_url())

    def download_html(self):
        if False:
            while True:
                i = 10
        url = self.pyfile.url
        self.log_debug(f'Requesting page: {url}')
        self.data = self.load(url)

    def file_exists(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.data:
            self.download_html()
        if '<title>Veehd</title>' in self.data:
            return False
        return True

    def get_file_name(self):
        if False:
            i = 10
            return i + 15
        if not self.data:
            self.download_html()
        m = re.search('<title.*?>(.+?) on Veehd</title>', self.data)
        if m is None:
            self.error(self._('Video title not found'))
        name = m.group(1)
        if self.config.get('filename_spaces'):
            pattern = '[^\\w ]+'
        else:
            pattern = '[^\\w.]+'
        return re.sub(pattern, self.config.get('replacement_char'), name) + '.avi'

    def get_file_url(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the absolute downloadable filepath.\n        '
        if not self.data:
            self.download_html()
        m = re.search('<embed type="video/divx" src="(http://([^/]*\\.)?veehd\\.com/dl/.+?)"', self.data)
        if m is None:
            self.error(self._('Embedded video url not found'))
        return m.group(1)