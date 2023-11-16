__author__ = 'Yaroslav Halchenko, Serg G. Brester (aka sebres)'
__copyright__ = 'Copyright (c) 2007 Yaroslav Halchenko, 2015 Serg G. Brester (aka sebres)'
__license__ = 'GPL'
import os
import re
import sys
from ..helpers import getLogger
from configparser import ConfigParser as SafeConfigParser, BasicInterpolation, InterpolationMissingOptionError, NoOptionError, NoSectionError

class BasicInterpolationWithName(BasicInterpolation):
    """Decorator to bring __name__ interpolation back.

	Original handling of __name__ was removed because of
	functional deficiencies: http://bugs.python.org/issue10489

	commit v3.2a4-105-g61f2761
	Author: Lukasz Langa <lukasz@langa.pl>
	Date:	Sun Nov 21 13:41:35 2010 +0000

	Issue #10489: removed broken `__name__` support from configparser

	But should be fine to reincarnate for our use case
	"""

    def _interpolate_some(self, parser, option, accum, rest, section, map, *args, **kwargs):
        if False:
            print('Hello World!')
        if section and (not __name__ in map):
            map = map.copy()
            map['__name__'] = section
            parser._map_section_options(section, option, rest, map)
            return super(BasicInterpolationWithName, self)._interpolate_some(parser, option, accum, rest, section, map, *args, **kwargs)

def _expandConfFilesWithLocal(filenames):
    if False:
        for i in range(10):
            print('nop')
    'Expands config files with local extension.\n\t'
    newFilenames = []
    for filename in filenames:
        newFilenames.append(filename)
        localname = os.path.splitext(filename)[0] + '.local'
        if localname not in filenames and os.path.isfile(localname):
            newFilenames.append(localname)
    return newFilenames
logSys = getLogger(__name__)
logLevel = 7
__all__ = ['SafeConfigParserWithIncludes']

class SafeConfigParserWithIncludes(SafeConfigParser):
    """
	Class adds functionality to SafeConfigParser to handle included
	other configuration files (or may be urls, whatever in the future)

	File should have section [includes] and only 2 options implemented
	are 'files_before' and 'files_after' where files are listed 1 per
	line.

	Example:

[INCLUDES]
before = 1.conf
         3.conf

after = 1.conf

	It is a simple implementation, so just basic care is taken about
	recursion. Includes preserve right order, ie new files are
	inserted to the list of read configs before original, and their
	includes correspondingly so the list should follow the leaves of
	the tree.

	I wasn't sure what would be the right way to implement generic (aka c++
	template) so we could base at any *configparser class... so I will
	leave it for the future

	"""
    SECTION_NAME = 'INCLUDES'
    SECTION_OPTNAME_CRE = re.compile('^([\\w\\-]+)/([^\\s>]+)$')
    SECTION_OPTSUBST_CRE = re.compile('%\\(([\\w\\-]+/([^\\)]+))\\)s')
    CONDITIONAL_RE = re.compile('^(\\w+)(\\?.+)$')
    if sys.version_info >= (3, 2):

        def __init__(self, share_config=None, *args, **kwargs):
            if False:
                return 10
            kwargs = kwargs.copy()
            kwargs['interpolation'] = BasicInterpolationWithName()
            kwargs['inline_comment_prefixes'] = ';'
            super(SafeConfigParserWithIncludes, self).__init__(*args, **kwargs)
            self._cfg_share = share_config
    else:

        def __init__(self, share_config=None, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            SafeConfigParser.__init__(self, *args, **kwargs)
            self._cfg_share = share_config

    def get_ex(self, section, option, raw=False, vars={}):
        if False:
            while True:
                i = 10
        'Get an option value for a given section.\n\t\t\n\t\tIn opposite to `get`, it differentiate session-related option name like `sec/opt`.\n\t\t'
        sopt = None
        if '/' in option:
            sopt = SafeConfigParserWithIncludes.SECTION_OPTNAME_CRE.search(option)
        if sopt:
            sec = sopt.group(1)
            opt = sopt.group(2)
            seclwr = sec.lower()
            if seclwr == 'known':
                sopt = ('KNOWN/' + section, section)
            else:
                sopt = (sec,) if seclwr != 'default' else ('DEFAULT',)
            for sec in sopt:
                try:
                    v = self.get(sec, opt, raw=raw)
                    return v
                except (NoSectionError, NoOptionError) as e:
                    pass
        v = self.get(section, option, raw=raw, vars=vars)
        return v

    def _map_section_options(self, section, option, rest, defaults):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\tInterpolates values of the section options (name syntax `%(section/option)s`).\n\n\t\tFallback: try to wrap missing default options as "default/options" resp. "known/options"\n\t\t'
        if '/' not in rest or '%(' not in rest:
            return 0
        rplcmnt = 0
        soptrep = SafeConfigParserWithIncludes.SECTION_OPTSUBST_CRE.findall(rest)
        if not soptrep:
            return 0
        for (sopt, opt) in soptrep:
            if sopt not in defaults:
                sec = sopt[:~len(opt)]
                seclwr = sec.lower()
                if seclwr != 'default':
                    usedef = 0
                    if seclwr == 'known':
                        try:
                            v = self._sections['KNOWN/' + section][opt]
                        except KeyError:
                            usedef = 1
                    else:
                        try:
                            try:
                                sec = self._sections[sec]
                            except KeyError:
                                continue
                            v = sec[opt]
                        except KeyError:
                            usedef = 1
                else:
                    usedef = 1
                if usedef:
                    try:
                        v = self._defaults[opt]
                    except KeyError:
                        continue
                rplcmnt = 1
                try:
                    defaults[sopt] = v
                except:
                    try:
                        defaults._maps[0][sopt] = v
                    except:
                        self._defaults[sopt] = v
        return rplcmnt

    @property
    def share_config(self):
        if False:
            for i in range(10):
                print('nop')
        return self._cfg_share

    def _getSharedSCPWI(self, filename):
        if False:
            for i in range(10):
                print('nop')
        SCPWI = SafeConfigParserWithIncludes
        if self._cfg_share:
            hashv = 'inc:' + (filename if not isinstance(filename, list) else '\x01'.join(filename))
            (cfg, i) = self._cfg_share.get(hashv, (None, None))
            if cfg is None:
                cfg = SCPWI(share_config=self._cfg_share)
                i = cfg.read(filename, get_includes=False)
                self._cfg_share[hashv] = (cfg, i)
            elif logSys.getEffectiveLevel() <= logLevel:
                logSys.log(logLevel, '    Shared file: %s', filename)
        else:
            cfg = SCPWI()
            i = cfg.read(filename, get_includes=False)
        return (cfg, i)

    def _getIncludes(self, filenames, seen=[]):
        if False:
            i = 10
            return i + 15
        if not isinstance(filenames, list):
            filenames = [filenames]
        filenames = _expandConfFilesWithLocal(filenames)
        if self._cfg_share:
            hashv = 'inc-path:' + '\x01'.join(filenames)
            fileNamesFull = self._cfg_share.get(hashv)
            if fileNamesFull is None:
                fileNamesFull = []
                for filename in filenames:
                    fileNamesFull += self.__getIncludesUncached(filename, seen)
                self._cfg_share[hashv] = fileNamesFull
            return fileNamesFull
        fileNamesFull = []
        for filename in filenames:
            fileNamesFull += self.__getIncludesUncached(filename, seen)
        return fileNamesFull

    def __getIncludesUncached(self, resource, seen=[]):
        if False:
            return 10
        '\n\t\tGiven 1 config resource returns list of included files\n\t\t(recursively) with the original one as well\n\t\tSimple loops are taken care about\n\t\t'
        SCPWI = SafeConfigParserWithIncludes
        try:
            (parser, i) = self._getSharedSCPWI(resource)
            if not i:
                return []
        except UnicodeDecodeError as e:
            logSys.error("Error decoding config file '%s': %s" % (resource, e))
            return []
        resourceDir = os.path.dirname(resource)
        newFiles = [('before', []), ('after', [])]
        if SCPWI.SECTION_NAME in parser.sections():
            for (option_name, option_list) in newFiles:
                if option_name in parser.options(SCPWI.SECTION_NAME):
                    newResources = parser.get(SCPWI.SECTION_NAME, option_name)
                    for newResource in newResources.split('\n'):
                        if os.path.isabs(newResource):
                            r = newResource
                        else:
                            r = os.path.join(resourceDir, newResource)
                        if r in seen:
                            continue
                        s = seen + [resource]
                        option_list += self._getIncludes(r, s)
        return newFiles[0][1] + [resource] + newFiles[1][1]

    def get_defaults(self):
        if False:
            while True:
                i = 10
        return self._defaults

    def get_sections(self):
        if False:
            while True:
                i = 10
        return self._sections

    def options(self, section, withDefault=True):
        if False:
            return 10
        'Return a list of option names for the given section name.\n\n\t\tParameter `withDefault` controls the include of names from section `[DEFAULT]`\n\t\t'
        try:
            opts = self._sections[section]
        except KeyError:
            raise NoSectionError(section)
        if withDefault:
            return set(opts.keys()) | set(self._defaults)
        return list(opts.keys())

    def read(self, filenames, get_includes=True):
        if False:
            i = 10
            return i + 15
        if not isinstance(filenames, list):
            filenames = [filenames]
        fileNamesFull = []
        if get_includes:
            fileNamesFull += self._getIncludes(filenames)
        else:
            fileNamesFull = filenames
        if not fileNamesFull:
            return []
        logSys.info('  Loading files: %s', fileNamesFull)
        if get_includes or len(fileNamesFull) > 1:
            ret = []
            alld = self.get_defaults()
            alls = self.get_sections()
            for filename in fileNamesFull:
                (cfg, i) = self._getSharedSCPWI(filename)
                if i:
                    ret += i
                    alld.update(cfg.get_defaults())
                    for (n, s) in cfg.get_sections().items():
                        cond = SafeConfigParserWithIncludes.CONDITIONAL_RE.match(n)
                        if cond:
                            (n, cond) = cond.groups()
                            s = s.copy()
                            try:
                                del s['__name__']
                            except KeyError:
                                pass
                            for k in list(s.keys()):
                                v = s.pop(k)
                                s[k + cond] = v
                        s2 = alls.get(n)
                        if isinstance(s2, dict):
                            self.merge_section('KNOWN/' + n, dict([i for i in iter(s2.items()) if i[0] in s]), '')
                            s2.update(s)
                        else:
                            alls[n] = s.copy()
            return ret
        if logSys.getEffectiveLevel() <= logLevel:
            logSys.log(logLevel, '    Reading file: %s', fileNamesFull[0])
        if sys.version_info >= (3, 2):
            return SafeConfigParser.read(self, fileNamesFull, encoding='utf-8')
        else:
            return SafeConfigParser.read(self, fileNamesFull)

    def merge_section(self, section, options, pref=None):
        if False:
            print('Hello World!')
        alls = self.get_sections()
        try:
            sec = alls[section]
        except KeyError:
            alls[section] = sec = dict()
        if not pref:
            sec.update(options)
            return
        sk = {}
        for (k, v) in options.items():
            if not k.startswith(pref) and k != '__name__':
                sk[pref + k] = v
        sec.update(sk)