__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
from .configreader import ConfigReader
from ..helpers import getLogger, str2LogLevel
logSys = getLogger(__name__)

class Fail2banReader(ConfigReader):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        ConfigReader.__init__(self, **kwargs)

    def read(self):
        if False:
            for i in range(10):
                print('nop')
        ConfigReader.read(self, 'fail2ban')

    def getEarlyOptions(self):
        if False:
            return 10
        opts = [['string', 'socket', '/var/run/fail2ban/fail2ban.sock'], ['string', 'pidfile', '/var/run/fail2ban/fail2ban.pid'], ['string', 'loglevel', 'INFO'], ['string', 'logtarget', '/var/log/fail2ban.log'], ['string', 'syslogsocket', 'auto']]
        return ConfigReader.getOptions(self, 'Definition', opts)

    def getOptions(self, updateMainOpt=None):
        if False:
            return 10
        opts = [['string', 'loglevel', 'INFO'], ['string', 'logtarget', 'STDERR'], ['string', 'syslogsocket', 'auto'], ['string', 'allowipv6', 'auto'], ['string', 'dbfile', '/var/lib/fail2ban/fail2ban.sqlite3'], ['int', 'dbmaxmatches', None], ['string', 'dbpurgeage', '1d']]
        self.__opts = ConfigReader.getOptions(self, 'Definition', opts)
        if updateMainOpt:
            self.__opts.update(updateMainOpt)
        str2LogLevel(self.__opts.get('loglevel', 0))
        opts = [['int', 'stacksize']]
        if self.has_section('Thread'):
            thopt = ConfigReader.getOptions(self, 'Thread', opts)
            if thopt:
                self.__opts['thread'] = thopt

    def convert(self):
        if False:
            for i in range(10):
                print('nop')
        order = {'thread': 0, 'syslogsocket': 11, 'loglevel': 12, 'logtarget': 13, 'allowipv6': 14, 'dbfile': 50, 'dbmaxmatches': 51, 'dbpurgeage': 51}
        stream = list()
        for opt in self.__opts:
            if opt in order:
                stream.append((order[opt], ['set', opt, self.__opts[opt]]))
        return [opt[1] for opt in sorted(stream)]