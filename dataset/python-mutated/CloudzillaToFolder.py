import re
from ..base.simple_decrypter import SimpleDecrypter

class CloudzillaToFolder(SimpleDecrypter):
    __name__ = 'CloudzillaToFolder'
    __type__ = 'decrypter'
    __version__ = '0.11'
    __status__ = 'testing'
    __pattern__ = 'http://(?:www\\.)?cloudzilla\\.to/share/folder/(?P<ID>[\\w^_]+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('folder_per_package', 'Default;Yes;No', 'Create folder for each package', 'Default'), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Cloudzilla.to folder decrypter plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com')]
    INFO_PATTERN = '<span class="name" title="(?P<N>.+?)"'
    OFFLINE_PATTERN = '>File not found...<'
    LINK_PATTERN = '<a href="(.+?)" class="item_href">'
    PASSWORD_PATTERN = '<div id="pwd_protected">'

    def check_errors(self):
        if False:
            print('Hello World!')
        m = re.search(self.PASSWORD_PATTERN, self.data)
        if m is not None:
            self.data = self.load(self.pyfile.url, get={'key': self.get_password()})
        if re.search(self.PASSWORD_PATTERN, self.data):
            self.retry(msg='Wrong password')