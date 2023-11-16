"""distutils.filelist

Provides the FileList class, used for poking about the filesystem
and building lists of files.
"""
import os, re
import fnmatch
import functools
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsInternalError
from distutils import log

class FileList:
    """A list of files built by on exploring the filesystem and filtered by
    applying various patterns to what we find there.

    Instance attributes:
      dir
        directory from which files will be taken -- only used if
        'allfiles' not supplied to constructor
      files
        list of filenames currently being built/filtered/manipulated
      allfiles
        complete list of files under consideration (ie. without any
        filtering applied)
    """

    def __init__(self, warn=None, debug_print=None):
        if False:
            for i in range(10):
                print('nop')
        self.allfiles = None
        self.files = []

    def set_allfiles(self, allfiles):
        if False:
            while True:
                i = 10
        self.allfiles = allfiles

    def findall(self, dir=os.curdir):
        if False:
            while True:
                i = 10
        self.allfiles = findall(dir)

    def debug_print(self, msg):
        if False:
            while True:
                i = 10
        "Print 'msg' to stdout if the global DEBUG (taken from the\n        DISTUTILS_DEBUG environment variable) flag is true.\n        "
        from distutils.debug import DEBUG
        if DEBUG:
            print(msg)

    def append(self, item):
        if False:
            return 10
        self.files.append(item)

    def extend(self, items):
        if False:
            i = 10
            return i + 15
        self.files.extend(items)

    def sort(self):
        if False:
            for i in range(10):
                print('nop')
        sortable_files = sorted(map(os.path.split, self.files))
        self.files = []
        for sort_tuple in sortable_files:
            self.files.append(os.path.join(*sort_tuple))

    def remove_duplicates(self):
        if False:
            print('Hello World!')
        for i in range(len(self.files) - 1, 0, -1):
            if self.files[i] == self.files[i - 1]:
                del self.files[i]

    def _parse_template_line(self, line):
        if False:
            i = 10
            return i + 15
        words = line.split()
        action = words[0]
        patterns = dir = dir_pattern = None
        if action in ('include', 'exclude', 'global-include', 'global-exclude'):
            if len(words) < 2:
                raise DistutilsTemplateError("'%s' expects <pattern1> <pattern2> ..." % action)
            patterns = [convert_path(w) for w in words[1:]]
        elif action in ('recursive-include', 'recursive-exclude'):
            if len(words) < 3:
                raise DistutilsTemplateError("'%s' expects <dir> <pattern1> <pattern2> ..." % action)
            dir = convert_path(words[1])
            patterns = [convert_path(w) for w in words[2:]]
        elif action in ('graft', 'prune'):
            if len(words) != 2:
                raise DistutilsTemplateError("'%s' expects a single <dir_pattern>" % action)
            dir_pattern = convert_path(words[1])
        else:
            raise DistutilsTemplateError("unknown action '%s'" % action)
        return (action, patterns, dir, dir_pattern)

    def process_template_line(self, line):
        if False:
            print('Hello World!')
        (action, patterns, dir, dir_pattern) = self._parse_template_line(line)
        if action == 'include':
            self.debug_print('include ' + ' '.join(patterns))
            for pattern in patterns:
                if not self.include_pattern(pattern, anchor=1):
                    log.warn("warning: no files found matching '%s'", pattern)
        elif action == 'exclude':
            self.debug_print('exclude ' + ' '.join(patterns))
            for pattern in patterns:
                if not self.exclude_pattern(pattern, anchor=1):
                    log.warn("warning: no previously-included files found matching '%s'", pattern)
        elif action == 'global-include':
            self.debug_print('global-include ' + ' '.join(patterns))
            for pattern in patterns:
                if not self.include_pattern(pattern, anchor=0):
                    log.warn("warning: no files found matching '%s' anywhere in distribution", pattern)
        elif action == 'global-exclude':
            self.debug_print('global-exclude ' + ' '.join(patterns))
            for pattern in patterns:
                if not self.exclude_pattern(pattern, anchor=0):
                    log.warn("warning: no previously-included files matching '%s' found anywhere in distribution", pattern)
        elif action == 'recursive-include':
            self.debug_print('recursive-include %s %s' % (dir, ' '.join(patterns)))
            for pattern in patterns:
                if not self.include_pattern(pattern, prefix=dir):
                    log.warn("warning: no files found matching '%s' under directory '%s'", pattern, dir)
        elif action == 'recursive-exclude':
            self.debug_print('recursive-exclude %s %s' % (dir, ' '.join(patterns)))
            for pattern in patterns:
                if not self.exclude_pattern(pattern, prefix=dir):
                    log.warn("warning: no previously-included files matching '%s' found under directory '%s'", pattern, dir)
        elif action == 'graft':
            self.debug_print('graft ' + dir_pattern)
            if not self.include_pattern(None, prefix=dir_pattern):
                log.warn("warning: no directories found matching '%s'", dir_pattern)
        elif action == 'prune':
            self.debug_print('prune ' + dir_pattern)
            if not self.exclude_pattern(None, prefix=dir_pattern):
                log.warn("no previously-included directories found matching '%s'", dir_pattern)
        else:
            raise DistutilsInternalError("this cannot happen: invalid action '%s'" % action)

    def include_pattern(self, pattern, anchor=1, prefix=None, is_regex=0):
        if False:
            i = 10
            return i + 15
        'Select strings (presumably filenames) from \'self.files\' that\n        match \'pattern\', a Unix-style wildcard (glob) pattern.  Patterns\n        are not quite the same as implemented by the \'fnmatch\' module: \'*\'\n        and \'?\'  match non-special characters, where "special" is platform-\n        dependent: slash on Unix; colon, slash, and backslash on\n        DOS/Windows; and colon on Mac OS.\n\n        If \'anchor\' is true (the default), then the pattern match is more\n        stringent: "*.py" will match "foo.py" but not "foo/bar.py".  If\n        \'anchor\' is false, both of these will match.\n\n        If \'prefix\' is supplied, then only filenames starting with \'prefix\'\n        (itself a pattern) and ending with \'pattern\', with anything in between\n        them, will match.  \'anchor\' is ignored in this case.\n\n        If \'is_regex\' is true, \'anchor\' and \'prefix\' are ignored, and\n        \'pattern\' is assumed to be either a string containing a regex or a\n        regex object -- no translation is done, the regex is just compiled\n        and used as-is.\n\n        Selected strings will be added to self.files.\n\n        Return True if files are found, False otherwise.\n        '
        files_found = False
        pattern_re = translate_pattern(pattern, anchor, prefix, is_regex)
        self.debug_print("include_pattern: applying regex r'%s'" % pattern_re.pattern)
        if self.allfiles is None:
            self.findall()
        for name in self.allfiles:
            if pattern_re.search(name):
                self.debug_print(' adding ' + name)
                self.files.append(name)
                files_found = True
        return files_found

    def exclude_pattern(self, pattern, anchor=1, prefix=None, is_regex=0):
        if False:
            i = 10
            return i + 15
        "Remove strings (presumably filenames) from 'files' that match\n        'pattern'.  Other parameters are the same as for\n        'include_pattern()', above.\n        The list 'self.files' is modified in place.\n        Return True if files are found, False otherwise.\n        "
        files_found = False
        pattern_re = translate_pattern(pattern, anchor, prefix, is_regex)
        self.debug_print("exclude_pattern: applying regex r'%s'" % pattern_re.pattern)
        for i in range(len(self.files) - 1, -1, -1):
            if pattern_re.search(self.files[i]):
                self.debug_print(' removing ' + self.files[i])
                del self.files[i]
                files_found = True
        return files_found

def _find_all_simple(path):
    if False:
        i = 10
        return i + 15
    "\n    Find all files under 'path'\n    "
    results = (os.path.join(base, file) for (base, dirs, files) in os.walk(path, followlinks=True) for file in files)
    return filter(os.path.isfile, results)

def findall(dir=os.curdir):
    if False:
        return 10
    "\n    Find all files under 'dir' and return the list of full filenames.\n    Unless dir is '.', return full filenames with dir prepended.\n    "
    files = _find_all_simple(dir)
    if dir == os.curdir:
        make_rel = functools.partial(os.path.relpath, start=dir)
        files = map(make_rel, files)
    return list(files)

def glob_to_re(pattern):
    if False:
        i = 10
        return i + 15
    'Translate a shell-like glob pattern to a regular expression; return\n    a string containing the regex.  Differs from \'fnmatch.translate()\' in\n    that \'*\' does not match "special characters" (which are\n    platform-specific).\n    '
    pattern_re = fnmatch.translate(pattern)
    sep = os.sep
    if os.sep == '\\':
        sep = '\\\\\\\\'
    escaped = '\\1[^%s]' % sep
    pattern_re = re.sub('((?<!\\\\)(\\\\\\\\)*)\\.', escaped, pattern_re)
    return pattern_re

def translate_pattern(pattern, anchor=1, prefix=None, is_regex=0):
    if False:
        while True:
            i = 10
    "Translate a shell-like wildcard pattern to a compiled regular\n    expression.  Return the compiled regex.  If 'is_regex' true,\n    then 'pattern' is directly compiled to a regex (if it's a string)\n    or just returned as-is (assumes it's a regex object).\n    "
    if is_regex:
        if isinstance(pattern, str):
            return re.compile(pattern)
        else:
            return pattern
    (start, _, end) = glob_to_re('_').partition('_')
    if pattern:
        pattern_re = glob_to_re(pattern)
        assert pattern_re.startswith(start) and pattern_re.endswith(end)
    else:
        pattern_re = ''
    if prefix is not None:
        prefix_re = glob_to_re(prefix)
        assert prefix_re.startswith(start) and prefix_re.endswith(end)
        prefix_re = prefix_re[len(start):len(prefix_re) - len(end)]
        sep = os.sep
        if os.sep == '\\':
            sep = '\\\\'
        pattern_re = pattern_re[len(start):len(pattern_re) - len(end)]
        pattern_re = '%s\\A%s%s.*%s%s' % (start, prefix_re, sep, pattern_re, end)
    elif anchor:
        pattern_re = '%s\\A%s' % (start, pattern_re[len(start):])
    return re.compile(pattern_re)