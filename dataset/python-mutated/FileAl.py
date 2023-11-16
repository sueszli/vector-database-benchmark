import re
from ..base.xfs_downloader import XFSDownloader

class FileAl(XFSDownloader):
    __name__ = 'FileAl'
    __type__ = 'downloader'
    __version__ = '0.01'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?file\\.al/\\w{12}'
    __description__ = 'File.al downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('igel', None)]
    PLUGIN_DOMAIN = 'file.al'
    LINK_PATTERN = ('direct link.*?<a [^>]*href="(.+?)".*?>Click here to download', re.MULTILINE | re.DOTALL)
    WAIT_PATTERN = 'countdown.*?seconds.*?(\\d+)'
    RECAPTCHA_PATTERN = 'g-recaptcha.*?sitekey=[\\"\']([^\\"]*)'
    PREMIUM_ONLY_PATTERN = '(?:[Pp]remium Users only|can download files up to.*only)'

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.multi_dl = self.premium
        self.resume_download = True