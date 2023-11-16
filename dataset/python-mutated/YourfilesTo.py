import re
import urllib.parse
from ..base.downloader import BaseDownloader

class YourfilesTo(BaseDownloader):
    __name__ = 'YourfilesTo'
    __type__ = 'downloader'
    __version__ = '0.28'
    __status__ = 'testing'
    __pattern__ = 'http://(?:www\\.)?yourfiles\\.(to|biz)/\\?d=\\w+'
    __config__ = [('enabled', 'bool', 'Activated', True)]
    __description__ = 'Youfiles.to downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('jeix', 'jeix@hasnomail.de'), ('skydancer', 'skydancer@hasnomail.de')]

    def process(self, pyfile):
        if False:
            i = 10
            return i + 15
        self.pyfile = pyfile
        self.prepare()
        self.download(self.get_file_url())

    def prepare(self):
        if False:
            print('Hello World!')
        if not self.file_exists():
            self.offline()
        self.pyfile.name = self.get_file_name()
        self.wait(self.get_waiting_time())

    def get_waiting_time(self):
        if False:
            return 10
        if not self.data:
            self.download_html()
        m = re.search('var zzipitime = (\\d+);', self.data)
        if m is not None:
            sec = int(m.group(1))
        else:
            sec = 0
        return sec

    def download_html(self):
        if False:
            return 10
        url = self.pyfile.url
        self.data = self.load(url)

    def get_file_url(self):
        if False:
            return 10
        '\n        Returns the absolute downloadable filepath.\n        '
        url = re.search("var bla = '(.*?)';", self.data)
        if url:
            url = url.group(1)
            url = urllib.parse.unquote(url.replace('http://http:/http://', 'http://').replace('dumdidum', ''))
            return url
        else:
            self.error(self._('Absolute filepath not found'))

    def get_file_name(self):
        if False:
            i = 10
            return i + 15
        if not self.data:
            self.download_html()
        return re.search('<title>(.*)</title>', self.data).group(1)

    def file_exists(self):
        if False:
            return 10
        '\n        Returns True or False.\n        '
        if not self.data:
            self.download_html()
        if re.search('HTTP Status 404', self.data):
            return False
        else:
            return True