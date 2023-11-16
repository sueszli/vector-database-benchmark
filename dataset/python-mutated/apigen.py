"""
Attempt to generate templates for module reference with Sphinx.

To include extension modules, first identify them as valid in the
``_uri2path`` method, then handle them in the ``_parse_module_with_import``
script.

Notes
-----
This parsing is based on import and introspection of modules.
Previously functions and classes were found by parsing the text of .py files.

Extension modules should be discovered and included as well.

This is a modified version of a script originally shipped with the PyMVPA
project, then adapted for use first in NIPY and then in skimage. PyMVPA
is an MIT-licensed project.
"""
import os
import re
from types import BuiltinFunctionType, FunctionType, ModuleType
DEBUG = True

class ApiDocWriter:
    """Automatic detection and parsing of API docs to Sphinx-parsable reST format."""
    rst_section_levels = ['*', '=', '-', '~', '^']

    def __init__(self, package_name, rst_extension='.rst', package_skip_patterns=None, module_skip_patterns=None):
        if False:
            return 10
        "Initialize package for parsing\n\n        Parameters\n        ----------\n        package_name : string\n            Name of the top-level package. *package_name* must be the\n            name of an importable package.\n        rst_extension : string, optional\n            Extension for reST files, default '.rst'.\n        package_skip_patterns : None or sequence of {strings, regexps}\n            Sequence of strings giving URIs of packages to be excluded\n            Operates on the package path, starting at (including) the\n            first dot in the package path, after *package_name* - so,\n            if *package_name* is ``sphinx``, then ``sphinx.util`` will\n            result in ``.util`` being passed for searching by these\n            regexps.  If is None, gives default. Default is ``['\\.tests$']``.\n        module_skip_patterns : None or sequence\n            Sequence of strings giving URIs of modules to be excluded\n            Operates on the module name including preceding URI path,\n            back to the first dot after *package_name*.  For example\n            ``sphinx.util.console`` results in the string to search of\n            ``.util.console``.\n            If is None, gives default. Default is ``['\\.setup$', '\\._']``.\n        "
        if package_skip_patterns is None:
            package_skip_patterns = ['\\.tests$']
        if module_skip_patterns is None:
            module_skip_patterns = ['\\.setup$', '\\._']
        self.package_name = package_name
        self.rst_extension = rst_extension
        self.package_skip_patterns = package_skip_patterns
        self.module_skip_patterns = module_skip_patterns

    def get_package_name(self):
        if False:
            print('Hello World!')
        return self._package_name

    def set_package_name(self, package_name):
        if False:
            i = 10
            return i + 15
        "Set package_name\n\n        >>> docwriter = ApiDocWriter('sphinx')\n        >>> import sphinx\n        >>> docwriter.root_path == sphinx.__path__[0]\n        True\n        >>> docwriter.package_name = 'docutils'\n        >>> import docutils\n        >>> docwriter.root_path == docutils.__path__[0]\n        True\n        "
        self._package_name = package_name
        root_module = self._import(package_name)
        self.root_path = root_module.__path__[-1]
        self.written_modules = None
    package_name = property(get_package_name, set_package_name, None, 'get/set package_name')

    def _import(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Import namespace package.'
        mod = __import__(name)
        components = name.split('.')
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def _get_object_name(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Get second token in line.\n\n        >>> docwriter = ApiDocWriter(\'sphinx\')\n        >>> docwriter._get_object_name("  def func():  ")\n        \'func\'\n        >>> docwriter._get_object_name("  class Klass(object):  ")\n        \'Klass\'\n        >>> docwriter._get_object_name("  class Klass:  ")\n        \'Klass\'\n        '
        name = line.split()[1].split('(')[0].strip()
        return name.rstrip(':')

    def _uri2path(self, uri):
        if False:
            return 10
        "Convert uri to absolute filepath.\n\n        Parameters\n        ----------\n        uri : string\n            URI of python module to return path for\n\n        Returns\n        -------\n        path : None or string\n            Returns None if there is no valid path for this URI\n            Otherwise returns absolute file system path for URI\n\n        Examples\n        --------\n        >>> docwriter = ApiDocWriter('sphinx')\n        >>> import sphinx\n        >>> modpath = sphinx.__path__[0]\n        >>> res = docwriter._uri2path('sphinx.builder')\n        >>> res == os.path.join(modpath, 'builder.py')\n        True\n        >>> res = docwriter._uri2path('sphinx')\n        >>> res == os.path.join(modpath, '__init__.py')\n        True\n        >>> docwriter._uri2path('sphinx.does_not_exist')\n\n        "
        if uri == self.package_name:
            return os.path.join(self.root_path, '__init__.py')
        path = uri.replace(self.package_name + '.', '')
        path = path.replace('.', os.path.sep)
        path = os.path.join(self.root_path, path)
        if os.path.exists(path + '.py'):
            path += '.py'
        elif os.path.exists(os.path.join(path, '__init__.py')):
            path = os.path.join(path, '__init__.py')
        else:
            return None
        return path

    def _path2uri(self, dirpath):
        if False:
            i = 10
            return i + 15
        'Convert directory path to uri.'
        package_dir = self.package_name.replace('.', os.path.sep)
        relpath = dirpath.replace(self.root_path, package_dir)
        if relpath.startswith(os.path.sep):
            relpath = relpath[1:]
        return relpath.replace(os.path.sep, '.')

    def _parse_module(self, uri):
        if False:
            return 10
        'Parse module defined in uri.'
        filename = self._uri2path(uri)
        if filename is None:
            print(filename, 'erk')
            return ([], [])
        with open(filename) as f:
            (functions, classes) = self._parse_lines(f)
        return (functions, classes)

    def _parse_module_with_import(self, uri):
        if False:
            return 10
        'Look for functions and classes in the importable module.\n\n        Parameters\n        ----------\n        uri : str\n            The name of the module to be parsed. This module needs to be\n            importable.\n\n        Returns\n        -------\n        functions : list of str\n            A list of (public) function names in the module.\n        classes : list of str\n            A list of (public) class names in the module.\n        submodules : list of str\n            A list of (public) submodule names in the module.\n        '
        mod = __import__(uri, fromlist=[uri.split('.')[-1]])
        obj_strs = getattr(mod, '__all__', [obj for obj in dir(mod) if not obj.startswith('_')])
        functions = []
        classes = []
        submodules = []
        for obj_str in obj_strs:
            try:
                obj = getattr(mod, obj_str)
            except AttributeError:
                continue
            if isinstance(obj, FunctionType | BuiltinFunctionType):
                functions.append(obj_str)
            elif isinstance(obj, ModuleType) and 'skimage' in mod.__name__:
                submodules.append(obj_str)
            else:
                try:
                    issubclass(obj, object)
                    classes.append(obj_str)
                except TypeError:
                    pass
        return (functions, classes, submodules)

    def _parse_lines(self, linesource):
        if False:
            while True:
                i = 10
        'Parse lines of text for functions and classes.'
        functions = []
        classes = []
        for line in linesource:
            if line.startswith('def ') and line.count('('):
                name = self._get_object_name(line)
                if not name.startswith('_'):
                    functions.append(name)
            elif line.startswith('class '):
                name = self._get_object_name(line)
                if not name.startswith('_'):
                    classes.append(name)
            else:
                pass
        functions.sort()
        classes.sort()
        return (functions, classes)

    def generate_api_doc(self, uri):
        if False:
            for i in range(10):
                print('nop')
        "Make autodoc documentation template string for a module.\n\n        Parameters\n        ----------\n        uri : string\n            Python location of module - e.g 'sphinx.builder'.\n\n        Returns\n        -------\n        S : string\n            Contents of API doc.\n        "
        (functions, classes, submodules) = self._parse_module_with_import(uri)
        if not (len(functions) or len(classes) or len(submodules)) and DEBUG:
            print('WARNING: Empty -', uri)
            return ''
        functions = sorted(functions)
        classes = sorted(classes)
        submodules = sorted(submodules)
        ad = '.. AUTO-GENERATED FILE -- DO NOT EDIT!\n\n'
        title = ':mod:`' + uri + '`'
        ad += title + '\n' + self.rst_section_levels[1] * len(title) + '\n\n'
        ad += '.. automodule:: ' + uri + '\n\n'
        ad += '.. currentmodule:: ' + uri + '\n\n'
        ad += '.. autosummary::\n   :nosignatures:\n\n'
        for f in functions:
            ad += '   ' + uri + '.' + f + '\n'
        ad += '\n'
        for c in classes:
            ad += '   ' + uri + '.' + c + '\n'
        ad += '\n'
        for m in submodules:
            ad += '   ' + uri + '.' + m + '\n'
        ad += '\n'
        for f in functions:
            ad += '------------\n\n'
            full_f = uri + '.' + f
            ad += '\n.. autofunction:: ' + full_f + '\n\n'
            ad += f'    .. minigallery:: {full_f}\n\n'
        for c in classes:
            ad += '\n.. autoclass:: ' + c + '\n'
            ad += '  :members:\n  :inherited-members:\n  :undoc-members:\n  :show-inheritance:\n\n  .. automethod:: __init__\n\n'
            full_c = uri + '.' + c
            ad += f'    .. minigallery:: {full_c}\n\n'
        return ad

    def _survives_exclude(self, matchstr, match_type):
        if False:
            return 10
        "Return True if matchstr does not match patterns.\n\n        Removes ``self.package_name`` from the beginning of the string if present.\n\n        Examples\n        --------\n        >>> dw = ApiDocWriter('sphinx')\n        >>> dw._survives_exclude('sphinx.okpkg', 'package')\n        True\n        >>> dw.package_skip_patterns.append('^\\.badpkg$')\n        >>> dw._survives_exclude('sphinx.badpkg', 'package')\n        False\n        >>> dw._survives_exclude('sphinx.badpkg', 'module')\n        True\n        >>> dw._survives_exclude('sphinx.badmod', 'module')\n        True\n        >>> dw.module_skip_patterns.append('^\\.badmod$')\n        >>> dw._survives_exclude('sphinx.badmod', 'module')\n        False\n        "
        if match_type == 'module':
            patterns = self.module_skip_patterns
        elif match_type == 'package':
            patterns = self.package_skip_patterns
        else:
            raise ValueError(f'Cannot interpret match type "{match_type}"')
        L = len(self.package_name)
        if matchstr[:L] == self.package_name:
            matchstr = matchstr[L:]
        for pat in patterns:
            try:
                pat.search
            except AttributeError:
                pat = re.compile(pat)
            if pat.search(matchstr):
                return False
        return True

    def discover_modules(self):
        if False:
            print('Hello World!')
        "Return module sequence discovered from ``self.package_name``.\n\n        Returns\n        -------\n        mods : sequence\n            Sequence of module names within ``self.package_name``.\n\n        Examples\n        --------\n        >>> dw = ApiDocWriter('sphinx')\n        >>> mods = dw.discover_modules()\n        >>> 'sphinx.util' in mods\n        True\n        >>> dw.package_skip_patterns.append('\\.util$')\n        >>> 'sphinx.util' in dw.discover_modules()\n        False\n        >>>\n        "
        modules = [self.package_name]
        for (dirpath, dirnames, filenames) in os.walk(self.root_path):
            root_uri = self._path2uri(os.path.join(self.root_path, dirpath))
            for dirname in dirnames[:]:
                package_uri = '.'.join((root_uri, dirname))
                if self._uri2path(package_uri) and self._survives_exclude(package_uri, 'package'):
                    modules.append(package_uri)
                else:
                    dirnames.remove(dirname)
        return sorted(modules)

    def write_modules_api(self, modules, outdir):
        if False:
            while True:
                i = 10
        written_modules = []
        public_modules = [m for m in modules if not m.split('.')[-1].startswith('_')]
        for m in public_modules:
            api_str = self.generate_api_doc(m)
            if not api_str:
                continue
            outfile = os.path.join(outdir, m + self.rst_extension)
            with open(outfile, 'w') as fileobj:
                fileobj.write(api_str)
            written_modules.append(m)
        self.written_modules = written_modules

    def write_api_docs(self, outdir):
        if False:
            return 10
        'Generate API reST files.\n\n        Parameters\n        ----------\n        outdir : string\n            Directory name in which to store the files. Filenames for each module\n            are automatically created.\n\n        Notes\n        -----\n        Sets self.written_modules to list of written modules.\n        '
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        modules = self.discover_modules()
        self.write_modules_api(modules, outdir)

    def write_index(self, outdir, froot='gen', relative_to=None):
        if False:
            return 10
        "Make a reST API index file from the written files.\n\n        Parameters\n        ----------\n        outdir : string\n            Directory to which to write generated index file.\n        froot : string, optional\n            Root (filename without extension) of filename to write to\n            Defaults to 'gen'. We add ``self.rst_extension``.\n        relative_to : string\n            Path to which written filenames are relative. This\n            component of the written file path will be removed from\n            outdir, in the generated index. Default is None, meaning,\n            leave path as it is.\n        "
        if self.written_modules is None:
            raise ValueError('No modules written')
        path = os.path.join(outdir, froot + self.rst_extension)
        if relative_to is not None:
            relpath = (outdir + os.path.sep).replace(relative_to + os.path.sep, '')
        else:
            relpath = outdir
        print('outdir: ', relpath)
        with open(path, 'w') as idx:
            w = idx.write
            w('.. AUTO-GENERATED FILE -- DO NOT EDIT!\n\n')
            title = 'API reference'
            w(title + '\n')
            w('=' * len(title) + '\n\n')
            w('.. toctree::\n')
            w('   :maxdepth: 1\n\n')
            for f in self.written_modules:
                w(f'   {os.path.join(relpath, f)}\n\n')
            w('----------------------\n\n')
            w('.. toctree::\n')
            w('   :maxdepth: 1\n\n')
            w('   ../license\n')