import re
import urllib.parse
from ..base.decrypter import BaseDecrypter

class FourChanOrg(BaseDecrypter):
    __name__ = 'FourChanOrg'
    __type__ = 'decrypter'
    __version__ = '0.38'
    __status__ = 'testing'
    __pattern__ = 'http://(?:www\\.)?boards\\.4chan\\.org/\\w+/res/(\\d+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('folder_per_package', 'Default;Yes;No', 'Create folder for each package', 'Default')]
    __description__ = '4chan.org folder decrypter plugin'
    __license__ = 'GPLv3'
    __authors__ = []

    def decrypt(self, pyfile):
        if False:
            i = 10
            return i + 15
        pagehtml = self.load(pyfile.url)
        images = set(re.findall('(images\\.4chan\\.org/[^/]*/src/[^"<]+)', pagehtml))
        self.links = [urllib.parse.urljoin('http://', image) for image in images]