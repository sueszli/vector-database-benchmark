"""Routines for finding the sources that mypy will check"""
from __future__ import annotations
import functools
import os
from typing import Final, Sequence
from mypy.fscache import FileSystemCache
from mypy.modulefinder import PYTHON_EXTENSIONS, BuildSource, matches_exclude, mypy_path
from mypy.options import Options
PY_EXTENSIONS: Final = tuple(PYTHON_EXTENSIONS)

class InvalidSourceList(Exception):
    """Exception indicating a problem in the list of sources given to mypy."""

def create_source_list(paths: Sequence[str], options: Options, fscache: FileSystemCache | None=None, allow_empty_dir: bool=False) -> list[BuildSource]:
    if False:
        i = 10
        return i + 15
    'From a list of source files/directories, makes a list of BuildSources.\n\n    Raises InvalidSourceList on errors.\n    '
    fscache = fscache or FileSystemCache()
    finder = SourceFinder(fscache, options)
    sources = []
    for path in paths:
        path = os.path.normpath(path)
        if path.endswith(PY_EXTENSIONS):
            (name, base_dir) = finder.crawl_up(path)
            sources.append(BuildSource(path, name, None, base_dir))
        elif fscache.isdir(path):
            sub_sources = finder.find_sources_in_dir(path)
            if not sub_sources and (not allow_empty_dir):
                raise InvalidSourceList(f"There are no .py[i] files in directory '{path}'")
            sources.extend(sub_sources)
        else:
            mod = os.path.basename(path) if options.scripts_are_modules else None
            sources.append(BuildSource(path, mod, None))
    return sources

def keyfunc(name: str) -> tuple[bool, int, str]:
    if False:
        i = 10
        return i + 15
    'Determines sort order for directory listing.\n\n    The desirable properties are:\n    1) foo < foo.pyi < foo.py\n    2) __init__.py[i] < foo\n    '
    (base, suffix) = os.path.splitext(name)
    for (i, ext) in enumerate(PY_EXTENSIONS):
        if suffix == ext:
            return (base != '__init__', i, base)
    return (base != '__init__', -1, name)

def normalise_package_base(root: str) -> str:
    if False:
        print('Hello World!')
    if not root:
        root = os.curdir
    root = os.path.abspath(root)
    if root.endswith(os.sep):
        root = root[:-1]
    return root

def get_explicit_package_bases(options: Options) -> list[str] | None:
    if False:
        return 10
    'Returns explicit package bases to use if the option is enabled, or None if disabled.\n\n    We currently use MYPYPATH and the current directory as the package bases. In the future,\n    when --namespace-packages is the default could also use the values passed with the\n    --package-root flag, see #9632.\n\n    Values returned are normalised so we can use simple string comparisons in\n    SourceFinder.is_explicit_package_base\n    '
    if not options.explicit_package_bases:
        return None
    roots = mypy_path() + options.mypy_path + [os.getcwd()]
    return [normalise_package_base(root) for root in roots]

class SourceFinder:

    def __init__(self, fscache: FileSystemCache, options: Options) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.fscache = fscache
        self.explicit_package_bases = get_explicit_package_bases(options)
        self.namespace_packages = options.namespace_packages
        self.exclude = options.exclude
        self.verbosity = options.verbosity

    def is_explicit_package_base(self, path: str) -> bool:
        if False:
            print('Hello World!')
        assert self.explicit_package_bases
        return normalise_package_base(path) in self.explicit_package_bases

    def find_sources_in_dir(self, path: str) -> list[BuildSource]:
        if False:
            return 10
        sources = []
        seen: set[str] = set()
        names = sorted(self.fscache.listdir(path), key=keyfunc)
        for name in names:
            if name in ('__pycache__', 'site-packages', 'node_modules') or name.startswith('.'):
                continue
            subpath = os.path.join(path, name)
            if matches_exclude(subpath, self.exclude, self.fscache, self.verbosity >= 2):
                continue
            if self.fscache.isdir(subpath):
                sub_sources = self.find_sources_in_dir(subpath)
                if sub_sources:
                    seen.add(name)
                    sources.extend(sub_sources)
            else:
                (stem, suffix) = os.path.splitext(name)
                if stem not in seen and suffix in PY_EXTENSIONS:
                    seen.add(stem)
                    (module, base_dir) = self.crawl_up(subpath)
                    sources.append(BuildSource(subpath, module, None, base_dir))
        return sources

    def crawl_up(self, path: str) -> tuple[str, str]:
        if False:
            while True:
                i = 10
        'Given a .py[i] filename, return module and base directory.\n\n        For example, given "xxx/yyy/foo/bar.py", we might return something like:\n        ("foo.bar", "xxx/yyy")\n\n        If namespace packages is off, we crawl upwards until we find a directory without\n        an __init__.py\n\n        If namespace packages is on, we crawl upwards until the nearest explicit base directory.\n        Failing that, we return one past the highest directory containing an __init__.py\n\n        We won\'t crawl past directories with invalid package names.\n        The base directory returned is an absolute path.\n        '
        path = os.path.abspath(path)
        (parent, filename) = os.path.split(path)
        module_name = strip_py(filename) or filename
        (parent_module, base_dir) = self.crawl_up_dir(parent)
        if module_name == '__init__':
            return (parent_module, base_dir)
        module = module_join(parent_module, module_name)
        return (module, base_dir)

    def crawl_up_dir(self, dir: str) -> tuple[str, str]:
        if False:
            for i in range(10):
                print('nop')
        return self._crawl_up_helper(dir) or ('', dir)

    @functools.lru_cache
    def _crawl_up_helper(self, dir: str) -> tuple[str, str] | None:
        if False:
            return 10
        'Given a directory, maybe returns module and base directory.\n\n        We return a non-None value if we were able to find something clearly intended as a base\n        directory (as adjudicated by being an explicit base directory or by containing a package\n        with __init__.py).\n\n        This distinction is necessary for namespace packages, so that we know when to treat\n        ourselves as a subpackage.\n        '
        if self.explicit_package_bases is not None and self.is_explicit_package_base(dir):
            return ('', dir)
        (parent, name) = os.path.split(dir)
        if name.endswith('-stubs'):
            name = name[:-6]
        init_file = self.get_init_file(dir)
        if init_file is not None:
            if not name.isidentifier():
                raise InvalidSourceList(f'{name} is not a valid Python package name')
            (mod_prefix, base_dir) = self.crawl_up_dir(parent)
            return (module_join(mod_prefix, name), base_dir)
        if not name or not parent or (not name.isidentifier()):
            return None
        if not self.namespace_packages:
            return None
        result = self._crawl_up_helper(parent)
        if result is None:
            return None
        (mod_prefix, base_dir) = result
        return (module_join(mod_prefix, name), base_dir)

    def get_init_file(self, dir: str) -> str | None:
        if False:
            i = 10
            return i + 15
        "Check whether a directory contains a file named __init__.py[i].\n\n        If so, return the file's name (with dir prefixed).  If not, return None.\n\n        This prefers .pyi over .py (because of the ordering of PY_EXTENSIONS).\n        "
        for ext in PY_EXTENSIONS:
            f = os.path.join(dir, '__init__' + ext)
            if self.fscache.isfile(f):
                return f
            if ext == '.py' and self.fscache.init_under_package_root(f):
                return f
        return None

def module_join(parent: str, child: str) -> str:
    if False:
        print('Hello World!')
    'Join module ids, accounting for a possibly empty parent.'
    if parent:
        return parent + '.' + child
    return child

def strip_py(arg: str) -> str | None:
    if False:
        while True:
            i = 10
    'Strip a trailing .py or .pyi suffix.\n\n    Return None if no such suffix is found.\n    '
    for ext in PY_EXTENSIONS:
        if arg.endswith(ext):
            return arg[:-len(ext)]
    return None