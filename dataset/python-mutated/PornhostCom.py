import re
from ..base.downloader import BaseDownloader

class PornhostCom(BaseDownloader):
    __name__ = 'PornhostCom'
    __type__ = 'downloader'
    __version__ = '0.27'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?pornhost\\.com/\\d+'
    __config__ = [('enabled', 'bool', 'Activated', True)]
    __description__ = 'Pornhost.com downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('jeix', 'jeix@hasnomail.de'), ('GammaC0de', 'nitzo2001[AT}yahoo[DOT]com')]
    NAME_PATTERN = 'class="video-title">(.+?)<'
    LINK_PATTERN = '<source src="(https://cdn\\d+-dl.pornhost.com/.+)" type="video/mp4">'
    OFFLINE_PATTERN = '>Gallery not found<'

    def process(self, pyfile):
        if False:
            i = 10
            return i + 15
        self.data = self.load(pyfile.url)
        if re.search(self.OFFLINE_PATTERN, self.data) is not None:
            self.offline()
        m = re.search(self.NAME_PATTERN, self.data)
        if m is None:
            self.error(self._('name pattern not found'))
        pyfile.name = m.group(1) + '.mp4'
        m = re.search(self.LINK_PATTERN, self.data)
        if m is None:
            self.error(self._('link pattern not found'))
        self.download(m.group(1))