import re
from ..base.simple_decrypter import SimpleDecrypter

class Dereferer(SimpleDecrypter):
    __name__ = 'Dereferer'
    __type__ = 'decrypter'
    __version__ = '0.27'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?(?:\\w+\\.)*?(?P<DOMAIN>(?:[\\d.]+|[\\w\\-]{3,63}(?:\\.[a-zA-Z]{2,}){1,2})(?:\\:\\d+)?)/.*?(?P<LINK>[\\w^_]+://.+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('folder_per_package', 'Default;Yes;No', 'Create folder for each package', 'Default'), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Universal link dereferer'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com')]
    PLUGIN_DOMAIN = None
    PLUGIN_NAME = None
    DIRECT_LINK = False

    def _log(self, level, plugintype, pluginname, args, kwargs):
        if False:
            return 10
        args = (self.PLUGIN_NAME,) + args
        return super()._log(level, plugintype, pluginname, args, kwargs)

    def init(self):
        if False:
            return 10
        self.__pattern__ = self.pyload.plugin_manager.decrypter_plugins[self.classname]['pattern']
        self.PLUGIN_DOMAIN = re.match(self.__pattern__, self.pyfile.url).group('DOMAIN').lower()
        self.PLUGIN_NAME = ''.join((part.capitalize() for part in re.split('\\.|\\d+|-', self.PLUGIN_DOMAIN) if part != '.'))

    def get_links(self):
        if False:
            while True:
                i = 10
        return [self.info['pattern']['LINK']]