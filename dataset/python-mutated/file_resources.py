"""Interface to file resources.

This module provides functions for interfacing with files: opening, writing, and
querying.
"""
import codecs
import fnmatch
import os
import re
import sys
from configparser import ConfigParser
from tokenize import detect_encoding
from yapf.yapflib import errors
from yapf.yapflib import style
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
CR = '\r'
LF = '\n'
CRLF = '\r\n'

def _GetExcludePatternsFromYapfIgnore(filename):
    if False:
        while True:
            i = 10
    'Get a list of file patterns to ignore from .yapfignore.'
    ignore_patterns = []
    if os.path.isfile(filename) and os.access(filename, os.R_OK):
        with open(filename, 'r') as fd:
            for line in fd:
                if line.strip() and (not line.startswith('#')):
                    ignore_patterns.append(line.strip())
        if any((e.startswith('./') for e in ignore_patterns)):
            raise errors.YapfError('path in .yapfignore should not start with ./')
    return ignore_patterns

def _GetExcludePatternsFromPyprojectToml(filename):
    if False:
        while True:
            i = 10
    'Get a list of file patterns to ignore from pyproject.toml.'
    ignore_patterns = []
    if os.path.isfile(filename) and os.access(filename, os.R_OK):
        with open(filename, 'rb') as fd:
            pyproject_toml = tomllib.load(fd)
        ignore_patterns = pyproject_toml.get('tool', {}).get('yapfignore', {}).get('ignore_patterns', [])
        if any((e.startswith('./') for e in ignore_patterns)):
            raise errors.YapfError('path in pyproject.toml should not start with ./')
    return ignore_patterns

def GetExcludePatternsForDir(dirname):
    if False:
        return 10
    'Return patterns of files to exclude from ignorefile in a given directory.\n\n  Looks for .yapfignore in the directory dirname.\n\n  Arguments:\n    dirname: (unicode) The name of the directory.\n\n  Returns:\n    A List of file patterns to exclude if ignore file is found, otherwise empty\n    List.\n  '
    ignore_patterns = []
    yapfignore_file = os.path.join(dirname, '.yapfignore')
    if os.path.exists(yapfignore_file):
        ignore_patterns += _GetExcludePatternsFromYapfIgnore(yapfignore_file)
    pyproject_toml_file = os.path.join(dirname, 'pyproject.toml')
    if os.path.exists(pyproject_toml_file):
        ignore_patterns += _GetExcludePatternsFromPyprojectToml(pyproject_toml_file)
    return ignore_patterns

def GetDefaultStyleForDir(dirname, default_style=style.DEFAULT_STYLE):
    if False:
        return 10
    "Return default style name for a given directory.\n\n  Looks for .style.yapf or setup.cfg or pyproject.toml in the parent\n  directories.\n\n  Arguments:\n    dirname: (unicode) The name of the directory.\n    default_style: The style to return if nothing is found. Defaults to the\n                   global default style ('pep8') unless otherwise specified.\n\n  Returns:\n    The filename if found, otherwise return the default style.\n  "
    dirname = os.path.abspath(dirname)
    while True:
        style_file = os.path.join(dirname, style.LOCAL_STYLE)
        if os.path.exists(style_file):
            return style_file
        config_file = os.path.join(dirname, style.SETUP_CONFIG)
        try:
            fd = open(config_file)
        except IOError:
            pass
        else:
            with fd:
                config = ConfigParser()
                config.read_file(fd)
                if config.has_section('yapf'):
                    return config_file
        config_file = os.path.join(dirname, style.PYPROJECT_TOML)
        try:
            fd = open(config_file, 'rb')
        except IOError:
            pass
        else:
            with fd:
                pyproject_toml = tomllib.load(fd)
                style_dict = pyproject_toml.get('tool', {}).get('yapf', None)
                if style_dict is not None:
                    return config_file
        if not dirname or not os.path.basename(dirname) or dirname == os.path.abspath(os.path.sep):
            break
        dirname = os.path.dirname(dirname)
    global_file = os.path.expanduser(style.GLOBAL_STYLE)
    if os.path.exists(global_file):
        return global_file
    return default_style

def GetCommandLineFiles(command_line_file_list, recursive, exclude):
    if False:
        print('Hello World!')
    'Return the list of files specified on the command line.'
    return _FindPythonFiles(command_line_file_list, recursive, exclude)

def WriteReformattedCode(filename, reformatted_code, encoding='', in_place=False):
    if False:
        return 10
    'Emit the reformatted code.\n\n  Write the reformatted code into the file, if in_place is True. Otherwise,\n  write to stdout.\n\n  Arguments:\n    filename: (unicode) The name of the unformatted file.\n    reformatted_code: (unicode) The reformatted code.\n    encoding: (unicode) The encoding of the file.\n    in_place: (bool) If True, then write the reformatted code to the file.\n  '
    if in_place:
        with codecs.open(filename, mode='w', encoding=encoding) as fd:
            fd.write(reformatted_code)
    else:
        sys.stdout.buffer.write(reformatted_code.encode('utf-8'))

def LineEnding(lines):
    if False:
        return 10
    'Retrieve the line ending of the original source.'
    endings = {CRLF: 0, CR: 0, LF: 0}
    for line in lines:
        if line.endswith(CRLF):
            endings[CRLF] += 1
        elif line.endswith(CR):
            endings[CR] += 1
        elif line.endswith(LF):
            endings[LF] += 1
    return sorted((LF, CRLF, CR), key=endings.get, reverse=True)[0]

def _FindPythonFiles(filenames, recursive, exclude):
    if False:
        i = 10
        return i + 15
    'Find all Python files.'
    if exclude and any((e.startswith('./') for e in exclude)):
        raise errors.YapfError("path in '--exclude' should not start with ./")
    exclude = exclude and [e.rstrip('/' + os.path.sep) for e in exclude]
    python_files = []
    for filename in filenames:
        if filename != '.' and exclude and IsIgnored(filename, exclude):
            continue
        if os.path.isdir(filename):
            if not recursive:
                raise errors.YapfError("directory specified without '--recursive' flag: %s" % filename)
            excluded_dirs = []
            for (dirpath, dirnames, filelist) in os.walk(filename):
                if dirpath != '.' and exclude and IsIgnored(dirpath, exclude):
                    excluded_dirs.append(dirpath)
                    continue
                elif any((dirpath.startswith(e) for e in excluded_dirs)):
                    continue
                for f in filelist:
                    filepath = os.path.join(dirpath, f)
                    if exclude and IsIgnored(filepath, exclude):
                        continue
                    if IsPythonFile(filepath):
                        python_files.append(filepath)
                dirnames_ = [dirnames.pop(0) for i in range(len(dirnames))]
                for dirname in dirnames_:
                    dir_ = os.path.join(dirpath, dirname)
                    if IsIgnored(dir_, exclude):
                        excluded_dirs.append(dir_)
                    else:
                        dirnames.append(dirname)
        elif os.path.isfile(filename):
            python_files.append(filename)
    return python_files

def IsIgnored(path, exclude):
    if False:
        while True:
            i = 10
    'Return True if filename matches any patterns in exclude.'
    if exclude is None:
        return False
    path = path.lstrip(os.path.sep)
    while path.startswith('.' + os.path.sep):
        path = path[2:]
    return any((fnmatch.fnmatch(path, e.rstrip(os.path.sep)) for e in exclude))

def IsPythonFile(filename):
    if False:
        while True:
            i = 10
    'Return True if filename is a Python file.'
    if os.path.splitext(filename)[1] in frozenset({'.py', '.pyi'}):
        return True
    try:
        with open(filename, 'rb') as fd:
            encoding = detect_encoding(fd.readline)[0]
        with codecs.open(filename, mode='r', encoding=encoding) as fd:
            fd.read()
    except UnicodeDecodeError:
        encoding = 'latin-1'
    except (IOError, SyntaxError):
        return False
    try:
        with codecs.open(filename, mode='r', encoding=encoding) as fd:
            first_line = fd.readline(256)
    except IOError:
        return False
    return re.match('^#!.*\\bpython[23]?\\b', first_line)

def FileEncoding(filename):
    if False:
        for i in range(10):
            print('nop')
    "Return the file's encoding."
    with open(filename, 'rb') as fd:
        return detect_encoding(fd.readline)[0]