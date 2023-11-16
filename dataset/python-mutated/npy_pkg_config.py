import sys
import re
import os
from configparser import RawConfigParser
__all__ = ['FormatError', 'PkgNotFound', 'LibraryInfo', 'VariableSet', 'read_config', 'parse_flags']
_VAR = re.compile('\\$\\{([a-zA-Z0-9_-]+)\\}')

class FormatError(OSError):
    """
    Exception thrown when there is a problem parsing a configuration file.

    """

    def __init__(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.msg = msg

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.msg

class PkgNotFound(OSError):
    """Exception raised when a package can not be located."""

    def __init__(self, msg):
        if False:
            return 10
        self.msg = msg

    def __str__(self):
        if False:
            return 10
        return self.msg

def parse_flags(line):
    if False:
        i = 10
        return i + 15
    "\n    Parse a line from a config file containing compile flags.\n\n    Parameters\n    ----------\n    line : str\n        A single line containing one or more compile flags.\n\n    Returns\n    -------\n    d : dict\n        Dictionary of parsed flags, split into relevant categories.\n        These categories are the keys of `d`:\n\n        * 'include_dirs'\n        * 'library_dirs'\n        * 'libraries'\n        * 'macros'\n        * 'ignored'\n\n    "
    d = {'include_dirs': [], 'library_dirs': [], 'libraries': [], 'macros': [], 'ignored': []}
    flags = (' ' + line).split(' -')
    for flag in flags:
        flag = '-' + flag
        if len(flag) > 0:
            if flag.startswith('-I'):
                d['include_dirs'].append(flag[2:].strip())
            elif flag.startswith('-L'):
                d['library_dirs'].append(flag[2:].strip())
            elif flag.startswith('-l'):
                d['libraries'].append(flag[2:].strip())
            elif flag.startswith('-D'):
                d['macros'].append(flag[2:].strip())
            else:
                d['ignored'].append(flag)
    return d

def _escape_backslash(val):
    if False:
        while True:
            i = 10
    return val.replace('\\', '\\\\')

class LibraryInfo:
    """
    Object containing build information about a library.

    Parameters
    ----------
    name : str
        The library name.
    description : str
        Description of the library.
    version : str
        Version string.
    sections : dict
        The sections of the configuration file for the library. The keys are
        the section headers, the values the text under each header.
    vars : class instance
        A `VariableSet` instance, which contains ``(name, value)`` pairs for
        variables defined in the configuration file for the library.
    requires : sequence, optional
        The required libraries for the library to be installed.

    Notes
    -----
    All input parameters (except "sections" which is a method) are available as
    attributes of the same name.

    """

    def __init__(self, name, description, version, sections, vars, requires=None):
        if False:
            print('Hello World!')
        self.name = name
        self.description = description
        if requires:
            self.requires = requires
        else:
            self.requires = []
        self.version = version
        self._sections = sections
        self.vars = vars

    def sections(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the section headers of the config file.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        keys : list of str\n            The list of section headers.\n\n        '
        return list(self._sections.keys())

    def cflags(self, section='default'):
        if False:
            i = 10
            return i + 15
        val = self.vars.interpolate(self._sections[section]['cflags'])
        return _escape_backslash(val)

    def libs(self, section='default'):
        if False:
            return 10
        val = self.vars.interpolate(self._sections[section]['libs'])
        return _escape_backslash(val)

    def __str__(self):
        if False:
            while True:
                i = 10
        m = ['Name: %s' % self.name, 'Description: %s' % self.description]
        if self.requires:
            m.append('Requires:')
        else:
            m.append('Requires: %s' % ','.join(self.requires))
        m.append('Version: %s' % self.version)
        return '\n'.join(m)

class VariableSet:
    """
    Container object for the variables defined in a config file.

    `VariableSet` can be used as a plain dictionary, with the variable names
    as keys.

    Parameters
    ----------
    d : dict
        Dict of items in the "variables" section of the configuration file.

    """

    def __init__(self, d):
        if False:
            print('Hello World!')
        self._raw_data = dict([(k, v) for (k, v) in d.items()])
        self._re = {}
        self._re_sub = {}
        self._init_parse()

    def _init_parse(self):
        if False:
            i = 10
            return i + 15
        for (k, v) in self._raw_data.items():
            self._init_parse_var(k, v)

    def _init_parse_var(self, name, value):
        if False:
            i = 10
            return i + 15
        self._re[name] = re.compile('\\$\\{%s\\}' % name)
        self._re_sub[name] = value

    def interpolate(self, value):
        if False:
            i = 10
            return i + 15

        def _interpolate(value):
            if False:
                i = 10
                return i + 15
            for k in self._re.keys():
                value = self._re[k].sub(self._re_sub[k], value)
            return value
        while _VAR.search(value):
            nvalue = _interpolate(value)
            if nvalue == value:
                break
            value = nvalue
        return value

    def variables(self):
        if False:
            while True:
                i = 10
        '\n        Return the list of variable names.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        names : list of str\n            The names of all variables in the `VariableSet` instance.\n\n        '
        return list(self._raw_data.keys())

    def __getitem__(self, name):
        if False:
            i = 10
            return i + 15
        return self._raw_data[name]

    def __setitem__(self, name, value):
        if False:
            print('Hello World!')
        self._raw_data[name] = value
        self._init_parse_var(name, value)

def parse_meta(config):
    if False:
        for i in range(10):
            print('nop')
    if not config.has_section('meta'):
        raise FormatError('No meta section found !')
    d = dict(config.items('meta'))
    for k in ['name', 'description', 'version']:
        if not k in d:
            raise FormatError('Option %s (section [meta]) is mandatory, but not found' % k)
    if not 'requires' in d:
        d['requires'] = []
    return d

def parse_variables(config):
    if False:
        while True:
            i = 10
    if not config.has_section('variables'):
        raise FormatError('No variables section found !')
    d = {}
    for (name, value) in config.items('variables'):
        d[name] = value
    return VariableSet(d)

def parse_sections(config):
    if False:
        i = 10
        return i + 15
    return (meta_d, r)

def pkg_to_filename(pkg_name):
    if False:
        return 10
    return '%s.ini' % pkg_name

def parse_config(filename, dirs=None):
    if False:
        while True:
            i = 10
    if dirs:
        filenames = [os.path.join(d, filename) for d in dirs]
    else:
        filenames = [filename]
    config = RawConfigParser()
    n = config.read(filenames)
    if not len(n) >= 1:
        raise PkgNotFound('Could not find file(s) %s' % str(filenames))
    meta = parse_meta(config)
    vars = {}
    if config.has_section('variables'):
        for (name, value) in config.items('variables'):
            vars[name] = _escape_backslash(value)
    secs = [s for s in config.sections() if not s in ['meta', 'variables']]
    sections = {}
    requires = {}
    for s in secs:
        d = {}
        if config.has_option(s, 'requires'):
            requires[s] = config.get(s, 'requires')
        for (name, value) in config.items(s):
            d[name] = value
        sections[s] = d
    return (meta, vars, sections, requires)

def _read_config_imp(filenames, dirs=None):
    if False:
        print('Hello World!')

    def _read_config(f):
        if False:
            while True:
                i = 10
        (meta, vars, sections, reqs) = parse_config(f, dirs)
        for (rname, rvalue) in reqs.items():
            (nmeta, nvars, nsections, nreqs) = _read_config(pkg_to_filename(rvalue))
            for (k, v) in nvars.items():
                if not k in vars:
                    vars[k] = v
            for (oname, ovalue) in nsections[rname].items():
                if ovalue:
                    sections[rname][oname] += ' %s' % ovalue
        return (meta, vars, sections, reqs)
    (meta, vars, sections, reqs) = _read_config(filenames)
    if not 'pkgdir' in vars and 'pkgname' in vars:
        pkgname = vars['pkgname']
        if not pkgname in sys.modules:
            raise ValueError('You should import %s to get information on %s' % (pkgname, meta['name']))
        mod = sys.modules[pkgname]
        vars['pkgdir'] = _escape_backslash(os.path.dirname(mod.__file__))
    return LibraryInfo(name=meta['name'], description=meta['description'], version=meta['version'], sections=sections, vars=VariableSet(vars))
_CACHE = {}

def read_config(pkgname, dirs=None):
    if False:
        while True:
            i = 10
    "\n    Return library info for a package from its configuration file.\n\n    Parameters\n    ----------\n    pkgname : str\n        Name of the package (should match the name of the .ini file, without\n        the extension, e.g. foo for the file foo.ini).\n    dirs : sequence, optional\n        If given, should be a sequence of directories - usually including\n        the NumPy base directory - where to look for npy-pkg-config files.\n\n    Returns\n    -------\n    pkginfo : class instance\n        The `LibraryInfo` instance containing the build information.\n\n    Raises\n    ------\n    PkgNotFound\n        If the package is not found.\n\n    See Also\n    --------\n    misc_util.get_info, misc_util.get_pkg_info\n\n    Examples\n    --------\n    >>> npymath_info = np.distutils.npy_pkg_config.read_config('npymath')\n    >>> type(npymath_info)\n    <class 'numpy.distutils.npy_pkg_config.LibraryInfo'>\n    >>> print(npymath_info)\n    Name: npymath\n    Description: Portable, core math library implementing C99 standard\n    Requires:\n    Version: 0.1  #random\n\n    "
    try:
        return _CACHE[pkgname]
    except KeyError:
        v = _read_config_imp(pkg_to_filename(pkgname), dirs)
        _CACHE[pkgname] = v
        return v
if __name__ == '__main__':
    from optparse import OptionParser
    import glob
    parser = OptionParser()
    parser.add_option('--cflags', dest='cflags', action='store_true', help='output all preprocessor and compiler flags')
    parser.add_option('--libs', dest='libs', action='store_true', help='output all linker flags')
    parser.add_option('--use-section', dest='section', help='use this section instead of default for options')
    parser.add_option('--version', dest='version', action='store_true', help='output version')
    parser.add_option('--atleast-version', dest='min_version', help='Minimal version')
    parser.add_option('--list-all', dest='list_all', action='store_true', help='Minimal version')
    parser.add_option('--define-variable', dest='define_variable', help='Replace variable with the given value')
    (options, args) = parser.parse_args(sys.argv)
    if len(args) < 2:
        raise ValueError('Expect package name on the command line:')
    if options.list_all:
        files = glob.glob('*.ini')
        for f in files:
            info = read_config(f)
            print('%s\t%s - %s' % (info.name, info.name, info.description))
    pkg_name = args[1]
    d = os.environ.get('NPY_PKG_CONFIG_PATH')
    if d:
        info = read_config(pkg_name, ['numpy/_core/lib/npy-pkg-config', '.', d])
    else:
        info = read_config(pkg_name, ['numpy/_core/lib/npy-pkg-config', '.'])
    if options.section:
        section = options.section
    else:
        section = 'default'
    if options.define_variable:
        m = re.search('([\\S]+)=([\\S]+)', options.define_variable)
        if not m:
            raise ValueError('--define-variable option should be of the form --define-variable=foo=bar')
        else:
            name = m.group(1)
            value = m.group(2)
        info.vars[name] = value
    if options.cflags:
        print(info.cflags(section))
    if options.libs:
        print(info.libs(section))
    if options.version:
        print(info.version)
    if options.min_version:
        print(info.version >= options.min_version)