import pycurl
from ..base.decrypter import BaseDecrypter

class ShSt(BaseDecrypter):
    __name__ = 'ShSt'
    __type__ = 'decrypter'
    __version__ = '0.09'
    __status__ = 'testing'
    __pattern__ = 'http://sh\\.st/\\w+'
    __config__ = [('enabled', 'bool', 'Activated', True)]
    __description__ = 'Sh.St decrypter plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Frederik Möllers', 'fred-public@posteo.de')]
    NAME_PATTERN = '<title>(?P<N>.+?) -'

    def decrypt(self, pyfile):
        if False:
            i = 10
            return i + 15
        self.req.http.c.setopt(pycurl.USERAGENT, 'curl/7.42.1')
        header = self.load(self.pyfile.url, just_header=True, decode=False)
        target_url = header.get('location')
        self.links.append(target_url)