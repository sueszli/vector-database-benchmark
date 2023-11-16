import re
from ..base.downloader import BaseDownloader

class YoupornCom(BaseDownloader):
    __name__ = 'YoupornCom'
    __type__ = 'downloader'
    __version__ = '0.26'
    __status__ = 'testing'
    __pattern__ = 'http://(?:www\\.)?youporn\\.com/watch/.+'
    __config__ = [('enabled', 'bool', 'Activated', True)]
    __description__ = 'Youporn.com downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('willnix', 'willnix@pyload.net')]

    def process(self, pyfile):
        if False:
            while True:
                i = 10
        self.pyfile = pyfile
        if not self.file_exists():
            self.offline()
        pyfile.name = self.get_file_name()
        self.download(self.get_file_url())

    def download_html(self):
        if False:
            return 10
        url = self.pyfile.url
        self.data = self.load(url, post={'user_choice': 'Enter'}, cookies=False)

    def get_file_url(self):
        if False:
            while True:
                i = 10
        '\n        Returns the absolute downloadable filepath.\n        '
        if not self.data:
            self.download_html()
        return re.search('(http://download\\.youporn\\.com/download/\\d+\\?save=1)">', self.data).group(1)

    def get_file_name(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.data:
            self.download_html()
        file_name_pattern = '<title>(.+) - '
        return re.search(file_name_pattern, self.data).group(1).replace('&amp;', '&').replace('/', '') + '.flv'

    def file_exists(self):
        if False:
            return 10
        '\n        Returns True or False.\n        '
        if not self.data:
            self.download_html()
        if re.search('(.*invalid video_id.*)', self.data):
            return False
        else:
            return True