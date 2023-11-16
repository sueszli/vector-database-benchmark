__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import glob
import json
import os.path
import re
from .configreader import ConfigReaderUnshared, ConfigReader
from .filterreader import FilterReader
from .actionreader import ActionReader
from ..version import version
from ..helpers import getLogger, extractOptions, splitWithOptions, splitwords
logSys = getLogger(__name__)

class JailReader(ConfigReader):

    def __init__(self, name, force_enable=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ConfigReader.__init__(self, **kwargs)
        self.__name = name
        self.__filter = None
        self.__force_enable = force_enable
        self.__actions = list()
        self.__opts = None

    @property
    def options(self):
        if False:
            i = 10
            return i + 15
        return self.__opts

    def setName(self, value):
        if False:
            while True:
                i = 10
        self.__name = value

    def getName(self):
        if False:
            while True:
                i = 10
        return self.__name

    def read(self):
        if False:
            return 10
        out = ConfigReader.read(self, 'jail')
        if not self.__name in self.sections():
            raise ValueError('Jail %r was not found among available' % self.__name)
        return out

    def isEnabled(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__force_enable or (self.__opts and self.__opts.get('enabled', False))

    @staticmethod
    def _glob(path):
        if False:
            i = 10
            return i + 15
        'Given a path for glob return list of files to be passed to server.\n\n\t\tDangling symlinks are warned about and not returned\n\t\t'
        pathList = []
        for p in glob.glob(path):
            if os.path.exists(p):
                pathList.append(p)
            else:
                logSys.warning('File %s is a dangling link, thus cannot be monitored' % p)
        return pathList
    _configOpts1st = {'enabled': ['bool', False], 'backend': ['string', 'auto'], 'filter': ['string', '']}
    _configOpts = {'enabled': ['bool', False], 'backend': ['string', 'auto'], 'maxretry': ['int', None], 'maxmatches': ['int', None], 'findtime': ['string', None], 'bantime': ['string', None], 'bantime.increment': ['bool', None], 'bantime.factor': ['string', None], 'bantime.formula': ['string', None], 'bantime.multipliers': ['string', None], 'bantime.maxtime': ['string', None], 'bantime.rndtime': ['string', None], 'bantime.overalljails': ['bool', None], 'ignorecommand': ['string', None], 'ignoreself': ['bool', None], 'ignoreip': ['string', None], 'ignorecache': ['string', None], 'filter': ['string', ''], 'logtimezone': ['string', None], 'logencoding': ['string', None], 'logpath': ['string', None], 'action': ['string', '']}
    _configOpts.update(FilterReader._configOpts)
    _ignoreOpts = set(['action', 'filter', 'enabled'] + list(FilterReader._configOpts.keys()))

    def getOptions(self):
        if False:
            return 10
        basedir = self.getBaseDir()
        self.merge_defaults({'fail2ban_version': version, 'fail2ban_confpath': basedir})
        try:
            self.__opts = ConfigReader.getOptions(self, self.__name, self._configOpts1st, shouldExist=True)
            if not self.__opts:
                raise JailDefError('Init jail options failed')
            if not self.isEnabled():
                return True
            flt = self.__opts['filter']
            if flt:
                try:
                    (filterName, filterOpt) = extractOptions(flt)
                except ValueError as e:
                    raise JailDefError('Invalid filter definition %r: %s' % (flt, e))
                self.__filter = FilterReader(filterName, self.__name, filterOpt, share_config=self.share_config, basedir=basedir)
                ret = self.__filter.read()
                if not ret:
                    raise JailDefError('Unable to read the filter %r' % filterName)
                self.__filter.applyAutoOptions(self.__opts.get('backend', ''))
                self.__filter.getOptions(self.__opts, all=True)
                ConfigReader.merge_section(self, self.__name, self.__filter.getCombined(), 'known/')
            else:
                self.__filter = None
                logSys.warning('No filter set for jail %s' % self.__name)
            self.__opts = ConfigReader.getOptions(self, self.__name, self._configOpts)
            if not self.__opts:
                raise JailDefError('Read jail options failed')
            if self.__filter:
                self.__filter.getOptions(self.__opts)
            for act in splitWithOptions(self.__opts['action']):
                try:
                    act = act.strip()
                    if not act:
                        continue
                    try:
                        (actName, actOpt) = extractOptions(act)
                    except ValueError as e:
                        raise JailDefError('Invalid action definition %r: %s' % (act, e))
                    if actName.endswith('.py'):
                        self.__actions.append(['set', self.__name, 'addaction', actOpt.pop('actname', os.path.splitext(actName)[0]), os.path.join(basedir, 'action.d', actName), json.dumps(actOpt)])
                    else:
                        action = ActionReader(actName, self.__name, actOpt, share_config=self.share_config, basedir=basedir)
                        ret = action.read()
                        if ret:
                            action.getOptions(self.__opts)
                            self.__actions.append(action)
                        else:
                            raise JailDefError('Unable to read action %r' % actName)
                except JailDefError:
                    raise
                except Exception as e:
                    logSys.debug('Caught exception: %s', e, exc_info=True)
                    raise ValueError('Error in action definition %r: %r' % (act, e))
            if not len(self.__actions):
                logSys.warning('No actions were defined for %s' % self.__name)
        except JailDefError as e:
            e = str(e)
            logSys.error(e)
            if not self.__opts:
                self.__opts = dict()
            self.__opts['config-error'] = e
            return False
        return True

    def convert(self, allow_no_files=False):
        if False:
            i = 10
            return i + 15
        'Convert read before __opts to the commands stream\n\n\t\tParameters\n\t\t----------\n\t\tallow_missing : bool\n\t\t  Either to allow log files to be missing entirely.  Primarily is\n\t\t  used for testing\n\t\t '
        stream = []
        stream2 = []
        e = self.__opts.get('config-error')
        if e:
            stream.extend([['config-error', "Jail '%s' skipped, because of wrong configuration: %s" % (self.__name, e)]])
            return stream
        if self.__filter:
            stream.extend(self.__filter.convert())
        FilterReader._fillStream(stream, self.__opts, self.__name)
        for (opt, value) in self.__opts.items():
            if opt == 'logpath':
                if self.__opts.get('backend', '').startswith('systemd'):
                    continue
                found_files = 0
                for path in value.split('\n'):
                    path = path.rsplit(' ', 1)
                    (path, tail) = path if len(path) > 1 else (path[0], 'head')
                    pathList = JailReader._glob(path)
                    if len(pathList) == 0:
                        logSys.notice('No file(s) found for glob %s' % path)
                    for p in pathList:
                        found_files += 1
                        stream2.append(['set', self.__name, 'addlogpath', p, tail])
                if not found_files:
                    msg = 'Have not found any log file for %s jail' % self.__name
                    if not allow_no_files:
                        raise ValueError(msg)
                    logSys.warning(msg)
            elif opt == 'backend':
                backend = value
            elif opt == 'ignoreip':
                stream.append(['set', self.__name, 'addignoreip'] + splitwords(value))
            elif opt not in JailReader._ignoreOpts:
                stream.append(['set', self.__name, opt, value])
        if stream2:
            stream += stream2
        for action in self.__actions:
            if isinstance(action, (ConfigReaderUnshared, ConfigReader)):
                stream.extend(action.convert())
            else:
                stream.append(action)
        stream.insert(0, ['add', self.__name, backend])
        return stream

class JailDefError(Exception):
    pass