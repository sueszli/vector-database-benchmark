__author__ = 'Cyril Jaquier, Yaroslav Halchenko, Serg G. Brester (aka sebres)'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier, 2007 Yaroslav Halchenko, 2015 Serg G. Brester (aka sebres)'
__license__ = 'GPL'
import glob
import os
from configparser import NoOptionError, NoSectionError
from .configparserinc import sys, SafeConfigParserWithIncludes, logLevel
from ..helpers import getLogger, _as_bool, _merge_dicts, substituteRecursiveTags
logSys = getLogger(__name__)
CONVERTER = {'bool': _as_bool, 'int': int}

def _OptionsTemplateGen(options):
    if False:
        for i in range(10):
            print('nop')
    'Iterator over the options template with default options.\n\t\n\tEach options entry is composed of an array or tuple with:\n\t\t[[type, name, ?default?], ...]\n\tOr it is a dict:\n\t\t{name: [type, default], ...}\n\t'
    if isinstance(options, (list, tuple)):
        for optname in options:
            if len(optname) > 2:
                (opttype, optname, optvalue) = optname
            else:
                ((opttype, optname), optvalue) = (optname, None)
            yield (opttype, optname, optvalue)
    else:
        for optname in options:
            (opttype, optvalue) = options[optname]
            yield (opttype, optname, optvalue)

class ConfigReader:
    """Generic config reader class.

	A caching adapter which automatically reuses already shared configuration.
	"""

    def __init__(self, use_config=None, share_config=None, **kwargs):
        if False:
            while True:
                i = 10
        self._cfg_share = None
        self._cfg = None
        if use_config is not None:
            self._cfg = use_config
        if share_config is not None:
            self._cfg_share = share_config
            self._cfg_share_kwargs = kwargs
            self._cfg_share_basedir = None
        elif self._cfg is None:
            self._cfg = ConfigReaderUnshared(**kwargs)

    def setBaseDir(self, basedir):
        if False:
            print('Hello World!')
        if self._cfg:
            self._cfg.setBaseDir(basedir)
        else:
            self._cfg_share_basedir = basedir

    def getBaseDir(self):
        if False:
            while True:
                i = 10
        if self._cfg:
            return self._cfg.getBaseDir()
        else:
            return self._cfg_share_basedir

    @property
    def share_config(self):
        if False:
            while True:
                i = 10
        return self._cfg_share

    def read(self, name, once=True):
        if False:
            return 10
        " Overloads a default (not shared) read of config reader.\n\n\t  To prevent mutiple reads of config files with it includes, reads into \n\t  the config reader, if it was not yet cached/shared by 'name'.\n\t  "
        if not self._cfg:
            self._create_unshared(name)
        if once and self._cfg.read_cfg_files is not None:
            return self._cfg.read_cfg_files
        logSys.info('Loading configs for %s under %s ', name, self._cfg.getBaseDir())
        ret = self._cfg.read(name)
        self._cfg.read_cfg_files = ret
        return ret

    def _create_unshared(self, name=''):
        if False:
            while True:
                i = 10
        " Allocates and share a config file by it name.\n\n\t  Automatically allocates unshared or reuses shared handle by given 'name' and \n\t  init arguments inside a given shared storage.\n\t  "
        if not self._cfg and self._cfg_share is not None:
            self._cfg = self._cfg_share.get(name)
            if not self._cfg:
                self._cfg = ConfigReaderUnshared(share_config=self._cfg_share, **self._cfg_share_kwargs)
                if self._cfg_share_basedir is not None:
                    self._cfg.setBaseDir(self._cfg_share_basedir)
                self._cfg_share[name] = self._cfg
        else:
            self._cfg = ConfigReaderUnshared(**self._cfg_share_kwargs)

    def sections(self):
        if False:
            print('Hello World!')
        try:
            return (n for n in self._cfg.sections() if not n.startswith('KNOWN/'))
        except AttributeError:
            return []

    def has_section(self, sec):
        if False:
            return 10
        try:
            return self._cfg.has_section(sec)
        except AttributeError:
            return False

    def has_option(self, sec, opt, withDefault=True):
        if False:
            i = 10
            return i + 15
        return self._cfg.has_option(sec, opt) if withDefault else opt in self._cfg._sections.get(sec, {})

    def merge_defaults(self, d):
        if False:
            while True:
                i = 10
        self._cfg.get_defaults().update(d)

    def merge_section(self, section, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._cfg.merge_section(section, *args, **kwargs)
        except AttributeError:
            raise NoSectionError(section)

    def options(self, section, withDefault=False):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of option names for the given section name.\n\n\t\tParameter `withDefault` controls the include of names from section `[DEFAULT]`\n\t\t'
        try:
            return self._cfg.options(section, withDefault)
        except AttributeError:
            raise NoSectionError(section)

    def get(self, sec, opt, raw=False, vars={}):
        if False:
            return 10
        try:
            return self._cfg.get(sec, opt, raw=raw, vars=vars)
        except AttributeError:
            raise NoSectionError(sec)

    def getOptions(self, section, *args, **kwargs):
        if False:
            print('Hello World!')
        try:
            return self._cfg.getOptions(section, *args, **kwargs)
        except AttributeError:
            raise NoSectionError(section)

class ConfigReaderUnshared(SafeConfigParserWithIncludes):
    """Unshared config reader (previously ConfigReader).

	Do not use this class (internal not shared/cached represenation).
	Use ConfigReader instead.
	"""
    DEFAULT_BASEDIR = '/etc/fail2ban'

    def __init__(self, basedir=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        SafeConfigParserWithIncludes.__init__(self, *args, **kwargs)
        self.read_cfg_files = None
        self.setBaseDir(basedir)

    def setBaseDir(self, basedir):
        if False:
            i = 10
            return i + 15
        if basedir is None:
            basedir = ConfigReaderUnshared.DEFAULT_BASEDIR
        self._basedir = basedir.rstrip('/')

    def getBaseDir(self):
        if False:
            while True:
                i = 10
        return self._basedir

    def read(self, filename):
        if False:
            for i in range(10):
                print('nop')
        if not os.path.exists(self._basedir):
            raise ValueError('Base configuration directory %s does not exist ' % self._basedir)
        if filename.startswith('./'):
            filename = os.path.abspath(filename)
        basename = os.path.join(self._basedir, filename)
        logSys.debug('Reading configs for %s under %s ', filename, self._basedir)
        config_files = [basename + '.conf']
        config_dir = basename + '.d'
        config_files += sorted(glob.glob('%s/*.conf' % config_dir))
        config_files.append(basename + '.local')
        config_files += sorted(glob.glob('%s/*.local' % config_dir))
        config_files = list(filter(os.path.exists, config_files))
        if len(config_files):
            logSys.debug('Reading config files: %s', ', '.join(config_files))
            config_files_read = SafeConfigParserWithIncludes.read(self, config_files)
            missed = [cf for cf in config_files if cf not in config_files_read]
            if missed:
                logSys.error('Could not read config files: %s', ', '.join(missed))
            if config_files_read:
                return True
            logSys.error('Found no accessible config files for %r under %s', filename, self.getBaseDir())
            return False
        else:
            logSys.error('Found no accessible config files for %r ' % filename + ['under %s' % self.getBaseDir(), 'among existing ones: ' + ', '.join(config_files)][bool(len(config_files))])
            return False

    def getOptions(self, sec, options, pOptions=None, shouldExist=False, convert=True):
        if False:
            i = 10
            return i + 15
        values = dict()
        if pOptions is None:
            pOptions = {}
        for (opttype, optname, optvalue) in _OptionsTemplateGen(options):
            if optname in pOptions:
                continue
            try:
                v = self.get(sec, optname, vars=pOptions)
                values[optname] = v
                if convert:
                    conv = CONVERTER.get(opttype)
                    if conv:
                        if v is None:
                            continue
                        values[optname] = conv(v)
            except NoSectionError as e:
                if shouldExist:
                    raise
                logSys.error(e)
                values[optname] = optvalue
            except NoOptionError:
                if not optvalue is None:
                    logSys.debug("'%s' not defined in '%s'. Using default one: %r" % (optname, sec, optvalue))
                    values[optname] = optvalue
            except ValueError:
                logSys.warning("Wrong value for '" + optname + "' in '" + sec + "'. Using default one: '" + repr(optvalue) + "'")
                values[optname] = optvalue
        return values

class DefinitionInitConfigReader(ConfigReader):
    """Config reader for files with options grouped in [Definition] and
	[Init] sections.

	Is a base class for readers of filters and actions, where definitions
	in jails might provide custom values for options defined in [Init]
	section.
	"""
    _configOpts = []

    def __init__(self, file_, jailName, initOpts, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ConfigReader.__init__(self, **kwargs)
        if file_.startswith('./'):
            file_ = os.path.abspath(file_)
        self.setFile(file_)
        self.setJailName(jailName)
        self._initOpts = initOpts
        self._pOpts = dict()
        self._defCache = dict()

    def setFile(self, fileName):
        if False:
            return 10
        self._file = fileName
        self._initOpts = {}

    def getFile(self):
        if False:
            print('Hello World!')
        return self._file

    def setJailName(self, jailName):
        if False:
            for i in range(10):
                print('nop')
        self._jailName = jailName

    def getJailName(self):
        if False:
            while True:
                i = 10
        return self._jailName

    def read(self):
        if False:
            while True:
                i = 10
        return ConfigReader.read(self, self._file)

    def readexplicit(self):
        if False:
            while True:
                i = 10
        if not self._cfg:
            self._create_unshared(self._file)
        return SafeConfigParserWithIncludes.read(self._cfg, self._file)

    def getOptions(self, pOpts, all=False):
        if False:
            while True:
                i = 10
        if not pOpts:
            pOpts = dict()
        if self._initOpts:
            pOpts = _merge_dicts(pOpts, self._initOpts)
        self._opts = ConfigReader.getOptions(self, 'Definition', self._configOpts, pOpts, convert=False)
        self._pOpts = pOpts
        if self.has_section('Init'):
            getopt = lambda opt: self.get('Init', opt)
            for opt in self.options('Init', withDefault=False):
                if opt == '__name__':
                    continue
                v = None
                if not opt.startswith('known/'):
                    if v is None:
                        v = getopt(opt)
                    self._initOpts['known/' + opt] = v
                if opt not in self._initOpts:
                    if v is None:
                        v = getopt(opt)
                    self._initOpts[opt] = v
        if all and self.has_section('Definition'):
            for opt in self.options('Definition'):
                if opt == '__name__' or opt in self._opts:
                    continue
                self._opts[opt] = self.get('Definition', opt)

    def convertOptions(self, opts, configOpts):
        if False:
            i = 10
            return i + 15
        'Convert interpolated combined options to expected type.\n\t\t'
        for (opttype, optname, optvalue) in _OptionsTemplateGen(configOpts):
            conv = CONVERTER.get(opttype)
            if conv:
                v = opts.get(optname)
                if v is None:
                    continue
                try:
                    opts[optname] = conv(v)
                except ValueError:
                    logSys.warning('Wrong %s value %r for %r. Using default one: %r', opttype, v, optname, optvalue)
                    opts[optname] = optvalue

    def getCombOption(self, optname):
        if False:
            while True:
                i = 10
        'Get combined definition option (as string) using pre-set and init\n\t\toptions as preselection (values with higher precedence as specified in section).\n\n\t\tCan be used only after calling of getOptions.\n\t\t'
        try:
            return self._defCache[optname]
        except KeyError:
            try:
                v = self._cfg.get_ex('Definition', optname, vars=self._pOpts)
            except (NoSectionError, NoOptionError, ValueError):
                v = None
            self._defCache[optname] = v
            return v

    def getCombined(self, ignore=()):
        if False:
            for i in range(10):
                print('nop')
        combinedopts = self._opts
        if self._initOpts:
            combinedopts = _merge_dicts(combinedopts, self._initOpts)
        if not len(combinedopts):
            return {}
        ignore = set(ignore).copy()
        for n in combinedopts:
            cond = SafeConfigParserWithIncludes.CONDITIONAL_RE.match(n)
            if cond:
                (n, cond) = cond.groups()
                ignore.add(n)
        opts = substituteRecursiveTags(combinedopts, ignore=ignore, addrepl=self.getCombOption)
        if not opts:
            raise ValueError('recursive tag definitions unable to be resolved')
        self.convertOptions(opts, self._configOpts)
        return opts

    def convert(self):
        if False:
            print('Hello World!')
        raise NotImplementedError