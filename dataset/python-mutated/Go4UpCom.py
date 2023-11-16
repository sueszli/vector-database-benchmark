import json
import re
import urllib.parse
from ..base.simple_decrypter import SimpleDecrypter

class Go4UpCom(SimpleDecrypter):
    __name__ = 'Go4UpCom'
    __type__ = 'decrypter'
    __version__ = '0.19'
    __status__ = 'testing'
    __pattern__ = 'http://go4up\\.com/(dl/\\w{12}|rd/\\w{12}/\\d+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('folder_per_package', 'Default;Yes;No', 'Create folder for each package', 'Default'), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Go4Up.com decrypter plugin'
    __license__ = 'GPLv3'
    __authors__ = [('rlindner81', 'rlindner81@gmail.com'), ('Walter Purcaro', 'vuolter@gmail.com')]
    LINK_PATTERN = '(http://go4up\\.com/rd/.+?)<'
    NAME_PATTERN = '<title>Download (.+?)<'
    OFFLINE_PATTERN = '>\\s*(404 Page Not Found|File not Found|Mirror does not exist)'

    def get_links(self):
        if False:
            while True:
                i = 10
        links = []
        preference = self.config.get('preferred_hoster')
        hosterslink_re = re.search('(/download/gethosts/.+?)"', self.data)
        if hosterslink_re:
            hosters = self.load(urllib.parse.urljoin('http://go4up.com/', hosterslink_re.group(1)))
            for hoster in json.loads(hosters):
                if preference not in (0, int(hoster['hostId'])):
                    continue
                pagelink_re = re.search(self.LINK_PATTERN, hoster['link'])
                if pagelink_re:
                    page = self.load(pagelink_re.group(1))
                    link_re = re.search('<b><a href="(.+?)"', page)
                    if link_re:
                        links.append(link_re.group(1))
        return links