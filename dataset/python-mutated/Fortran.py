"""Dependency scanner for Fortran code."""
import re
import SCons.Node
import SCons.Node.FS
import SCons.Util
import SCons.Warnings
from . import Classic, Current, FindPathDirs

class F90Scanner(Classic):
    """
    A Classic Scanner subclass for Fortran source files which takes
    into account both USE and INCLUDE statements.  This scanner will
    work for both F77 and F90 (and beyond) compilers.

    Currently, this scanner assumes that the include files do not contain
    USE statements.  To enable the ability to deal with USE statements
    in include files, add logic right after the module names are found
    to loop over each include file, search for and locate each USE
    statement, and append each module name to the list of dependencies.
    Caching the search results in a common dictionary somewhere so that
    the same include file is not searched multiple times would be a
    smart thing to do.
    """

    def __init__(self, name, suffixes, path_variable, use_regex, incl_regex, def_regex, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.cre_use = re.compile(use_regex, re.M)
        self.cre_incl = re.compile(incl_regex, re.M)
        self.cre_def = re.compile(def_regex, re.M)

        def _scan(node, env, path, self=self):
            if False:
                return 10
            node = node.rfile()
            if not node.exists():
                return []
            return self.scan(node, env, path)
        kwargs['function'] = _scan
        kwargs['path_function'] = FindPathDirs(path_variable)
        kwargs['recursive'] = 1
        kwargs['skeys'] = suffixes
        kwargs['name'] = name
        Current.__init__(self, *args, **kwargs)

    def scan(self, node, env, path=()):
        if False:
            i = 10
            return i + 15
        if node.includes is not None:
            mods_and_includes = node.includes
        else:
            includes = self.cre_incl.findall(node.get_text_contents())
            modules = self.cre_use.findall(node.get_text_contents())
            defmodules = self.cre_def.findall(node.get_text_contents())
            d = {}
            for m in defmodules:
                d[m.lower()] = 1
            modules = [m for m in modules if m.lower() not in d]
            suffix = env.subst('$FORTRANMODSUFFIX')
            modules = [x.lower() + suffix for x in modules]
            mods_and_includes = SCons.Util.unique(includes + modules)
            node.includes = mods_and_includes
        nodes = []
        source_dir = node.get_dir()
        if callable(path):
            path = path()
        for dep in mods_and_includes:
            (n, i) = self.find_include(dep, source_dir, path)
            if n is None:
                SCons.Warnings.warn(SCons.Warnings.DependencyWarning, 'No dependency generated for file: %s (referenced by: %s) -- file not found' % (i, node))
            else:
                sortkey = self.sort_key(dep)
                nodes.append((sortkey, n))
        return [pair[1] for pair in sorted(nodes)]

def FortranScan(path_variable='FORTRANPATH'):
    if False:
        return 10
    'Return a prototype Scanner instance for scanning source files\n    for Fortran USE & INCLUDE statements'
    use_regex = '(?i)(?:^|;)\\s*USE(?:\\s+|(?:(?:\\s*,\\s*(?:NON_)?INTRINSIC)?\\s*::))\\s*(\\w+)'
    include_regex = '(?i)(?:^|[\'">]\\s*;)\\s*INCLUDE\\s+(?:\\w+_)?[<"\'](.+?)(?=["\'>])'
    def_regex = '(?i)^\\s*MODULE\\s+(?!PROCEDURE|SUBROUTINE|FUNCTION|PURE|ELEMENTAL)(\\w+)'
    scanner = F90Scanner('FortranScan', '$FORTRANSUFFIXES', path_variable, use_regex, include_regex, def_regex)
    return scanner