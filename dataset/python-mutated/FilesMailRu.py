import re
from ..base.downloader import BaseDownloader

class FilesMailRu(BaseDownloader):
    __name__ = 'FilesMailRu'
    __type__ = 'downloader'
    __version__ = '0.41'
    __status__ = 'testing'
    __pattern__ = 'http://(?:www\\.)?files\\.mail\\.ru/.+'
    __config__ = [('enabled', 'bool', 'Activated', True)]
    __description__ = 'Files.mail.ru downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('oZiRiz', 'ich@oziriz.de')]

    def setup(self):
        if False:
            return 10
        self.multi_dl = bool(self.account)

    def process(self, pyfile):
        if False:
            print('Hello World!')
        self.data = self.load(pyfile.url)
        self.url_pattern = '<a href="(.+?)" onclick="return Act\\(this\\, \\\'dlink\\\'\\, event\\)">(.+?)</a>'
        if '<div class="errorMessage mb10">' in self.data:
            self.offline()
        elif 'Page cannot be displayed' in self.data:
            self.offline()
        pyfile.name = self.get_file_name()
        if not self.account:
            self.prepare()
            self.download(self.get_file_url())
            self.my_post_process()
        else:
            self.download(self.get_file_url())
            self.my_post_process()

    def prepare(self):
        if False:
            return 10
        '\n        You have to wait some seconds.\n\n        Otherwise you will get a 40Byte HTML Page instead of the file you\n        expected\n        '
        self.wait(10)
        return True

    def get_file_url(self):
        if False:
            print('Hello World!')
        '\n        Gives you the URL to the file.\n\n        Extracted from the Files.mail.ru HTML-page stored in self.data\n        '
        return re.search(self.url_pattern, self.data).group(0).split('<a href="')[1].split('" onclick="return Act')[0]

    def get_file_name(self):
        if False:
            return 10
        '\n        Gives you the Name for each file.\n\n        Also extracted from the HTML-Page\n        '
        return re.search(self.url_pattern, self.data).group(0).split(', event)">')[1].split('</a>')[0]

    def my_post_process(self):
        if False:
            return 10
        if self.scan_download({'html': b'<meta name='}, read_size=50000) == 'html':
            self.log_info(self._('There was HTML Code in the Downloaded File ({})...redirect error? The Download will be restarted').format(self.pyfile.name))
            self.retry()