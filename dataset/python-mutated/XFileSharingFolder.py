import re
from ..base.xfs_decrypter import XFSDecrypter

class XFileSharingFolder(XFSDecrypter):
    __name__ = 'XFileSharingFolder'
    __type__ = 'decrypter'
    __version__ = '0.26'
    __status__ = 'testing'
    __pattern__ = '^unmatchable$'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('folder_per_package', 'Default;Yes;No', 'Create folder for each package', 'Default'), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'XFileSharing dummy folder decrypter plugin for addon'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com')]

    def _log(self, level, plugintype, pluginname, args, kwargs):
        if False:
            i = 10
            return i + 15
        args = (self.PLUGIN_NAME,) + args
        return super()._log(level, plugintype, pluginname, args, kwargs)

    def init(self):
        if False:
            return 10
        self.__pattern__ = self.pyload.plugin_manager.decrypter_plugins[self.classname]['pattern']
        self.PLUGIN_DOMAIN = re.match(self.__pattern__, self.pyfile.url).group('DOMAIN').lower()
        self.PLUGIN_NAME = ''.join((part.capitalize() for part in re.split('\\.|\\d+|-', self.PLUGIN_DOMAIN) if part != '.'))

    def setup_base(self):
        if False:
            print('Hello World!')
        super().setup_base()
        if self.account:
            self.req = self.pyload.request_factory.get_request(self.PLUGIN_NAME, self.account.user)
            self.premium = self.account.info['data']['premium']
        else:
            self.req = self.pyload.request_factory.get_request(self.classname)
            self.premium = False

    def load_account(self):
        if False:
            return 10
        class_name = self.classname
        self.__class__.__name__ = str(self.PLUGIN_NAME)
        super().load_account()
        self.__class__.__name__ = class_name