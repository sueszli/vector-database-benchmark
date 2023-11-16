__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
from .configreader import ConfigReader
from .jailreader import JailReader
from ..helpers import getLogger
logSys = getLogger(__name__)

class JailsReader(ConfigReader):

    def __init__(self, force_enable=False, **kwargs):
        if False:
            print('Hello World!')
        '\n\t\tParameters\n\t\t----------\n\t\tforce_enable : bool, optional\n\t\t  Passed to JailReader to force enable the jails.\n\t\t  It is for internal use\n\t\t'
        ConfigReader.__init__(self, **kwargs)
        self.__jails = list()
        self.__force_enable = force_enable

    @property
    def jails(self):
        if False:
            i = 10
            return i + 15
        return self.__jails

    def read(self):
        if False:
            return 10
        self.__jails = list()
        return ConfigReader.read(self, 'jail')

    def getOptions(self, section=None, ignoreWrong=True):
        if False:
            while True:
                i = 10
        'Reads configuration for jail(s) and adds enabled jails to __jails\n\t\t'
        opts = []
        self.__opts = ConfigReader.getOptions(self, 'Definition', opts)
        if section is None:
            sections = self.sections()
        else:
            sections = [section]
        parse_status = 0
        for sec in sections:
            if sec == 'INCLUDES':
                continue
            jail = JailReader(sec, force_enable=self.__force_enable, share_config=self.share_config, use_config=self._cfg)
            ret = jail.getOptions()
            if ret:
                if jail.isEnabled():
                    parse_status |= 1
                    self.__jails.append(jail)
            else:
                logSys.error('Errors in jail %r.%s', sec, ' Skipping...' if ignoreWrong else '')
                self.__jails.append(jail)
                parse_status |= 2
        return ignoreWrong and parse_status & 1 or not parse_status & 2

    def convert(self, allow_no_files=False):
        if False:
            while True:
                i = 10
        'Convert read before __opts and jails to the commands stream\n\n\t\tParameters\n\t\t----------\n\t\tallow_missing : bool\n\t\t  Either to allow log files to be missing entirely.  Primarily is\n\t\t  used for testing\n\t\t'
        stream = list()
        for jail in self.__jails:
            stream.extend(jail.convert(allow_no_files=allow_no_files))
        for jail in self.__jails:
            if not jail.options.get('config-error'):
                stream.append(['start', jail.getName()])
        return stream