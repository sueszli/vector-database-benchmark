""" Read/write source code from files.

Reading is tremendously more complex than one might think, due to encoding
issues and version differences of Python versions.
"""
import os
import re
import sys
from nuitka import Options, SourceCodeReferences
from nuitka.__past__ import unicode
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.plugins.Plugins import Plugins
from nuitka.PythonVersions import python_version, python_version_str
from nuitka.Tracing import general, my_print
from nuitka.utils.FileOperations import getFileContentByLine, putTextFileContents
from nuitka.utils.Shebang import getShebangFromSource, parseShebang
from nuitka.utils.Utils import isWin32OrPosixWindows
from .SyntaxErrors import raiseSyntaxError
_fstrings_installed = False

def _installFutureFStrings():
    if False:
        i = 10
        return i + 15
    'Install fake UTF8 handle just as future-fstrings does.'
    global _fstrings_installed
    if _fstrings_installed:
        return
    if python_version >= 864:
        import codecs
        try:
            codecs.lookup('future-fstrings')
        except LookupError:
            import encodings
            utf8 = encodings.search_function('utf8')
            codec_map = {'future-fstrings': utf8, 'future_fstrings': utf8}
            codecs.register(codec_map.get)
    else:
        try:
            import future_fstrings
        except ImportError:
            pass
        else:
            future_fstrings.register()
    _fstrings_installed = True

def _readSourceCodeFromFilename3(source_filename):
    if False:
        for i in range(10):
            print('nop')
    import tokenize
    _installFutureFStrings()
    with tokenize.open(source_filename) as source_file:
        return source_file.read()

def _detectEncoding2(source_file):
    if False:
        return 10
    encoding = 'ascii'
    line1 = source_file.readline()
    if line1.startswith(b'\xef\xbb\xbf'):
        encoding = 'utf-8'
    else:
        line1_match = re.search(b'coding[:=]\\s*([-\\w.]+)', line1)
        if line1_match:
            encoding = line1_match.group(1)
        else:
            line2 = source_file.readline()
            line2_match = re.search(b'coding[:=]\\s*([-\\w.]+)', line2)
            if line2_match:
                encoding = line2_match.group(1)
    source_file.seek(0)
    return encoding

def _readSourceCodeFromFilename2(source_filename):
    if False:
        while True:
            i = 10
    _installFutureFStrings()
    with open(source_filename, 'rU') as source_file:
        encoding = _detectEncoding2(source_file)
        source_code = source_file.read()
        if type(source_code) is not unicode and encoding == 'ascii':
            try:
                _source_code = source_code.decode(encoding)
            except UnicodeDecodeError as e:
                lines = source_code.split('\n')
                so_far = 0
                for (count, line) in enumerate(lines):
                    so_far += len(line) + 1
                    if so_far > e.args[2]:
                        break
                else:
                    count = -1
                wrong_byte = re.search('byte 0x([a-f0-9]{2}) in position', str(e)).group(1)
                raiseSyntaxError("Non-ASCII character '\\x%s' in file %s on line %d, but no encoding declared; see http://python.org/dev/peps/pep-0263/ for details" % (wrong_byte, source_filename, count + 1), SourceCodeReferences.fromFilename(source_filename).atLineNumber(count + 1), display_line=False)
    return source_code

def getSourceCodeDiff(source_code, source_code_modified):
    if False:
        print('Hello World!')
    import difflib
    diff = difflib.unified_diff(source_code.splitlines(), source_code_modified.splitlines(), 'original', 'modified', '', '', n=3)
    return list(diff)
_source_code_cache = {}

def readSourceCodeFromFilenameWithInformation(module_name, source_filename, pre_load=False):
    if False:
        for i in range(10):
            print('nop')
    key = (module_name, source_filename)
    if key in _source_code_cache:
        if pre_load:
            return _source_code_cache[key]
        else:
            return _source_code_cache.pop(key)
    if python_version < 768:
        source_code = _readSourceCodeFromFilename2(source_filename)
    else:
        source_code = _readSourceCodeFromFilename3(source_filename)
    if module_name is not None:
        (source_code_modified, contributing_plugins) = Plugins.onModuleSourceCode(module_name=module_name, source_filename=source_filename, source_code=source_code)
    else:
        source_code_modified = source_code
        contributing_plugins = ()
    if Options.shallShowSourceModifications() and source_code_modified != source_code:
        source_diff = getSourceCodeDiff(source_code, source_code_modified)
        if source_diff:
            my_print('%s:' % module_name.asString())
            for line in source_diff:
                my_print(line, end='\n' if not line.startswith('---') else '')
    result = (source_code_modified, source_code, contributing_plugins)
    if pre_load:
        _source_code_cache[key] = result
    return result

def readSourceCodeFromFilename(module_name, source_filename, pre_load=False):
    if False:
        i = 10
        return i + 15
    return readSourceCodeFromFilenameWithInformation(module_name=module_name, source_filename=source_filename, pre_load=pre_load)[0]

def checkPythonVersionFromCode(source_code):
    if False:
        return 10
    shebang = getShebangFromSource(source_code)
    if shebang is not None:
        (binary, _args) = parseShebang(shebang)
        if not isWin32OrPosixWindows():
            try:
                if os.path.samefile(sys.executable, binary):
                    return True
            except OSError:
                pass
        basename = os.path.basename(binary)
        if basename == 'python':
            result = python_version < 768
        elif basename == 'python3':
            result = python_version >= 768
        elif basename == 'python2':
            result = python_version < 768
        elif basename == 'python2.7':
            result = python_version < 768
        elif basename == 'python2.6':
            result = python_version < 624
        elif basename == 'python3.2':
            result = 816 > python_version >= 768
        elif basename == 'python3.3':
            result = 832 > python_version >= 816
        elif basename == 'python3.4':
            result = 848 > python_version >= 832
        elif basename == 'python3.5':
            result = 864 > python_version >= 848
        elif basename == 'python3.6':
            result = 880 > python_version >= 864
        elif basename == 'python3.7':
            result = 896 > python_version >= 880
        elif basename == 'python3.8':
            result = 912 > python_version >= 896
        elif basename == 'python3.9':
            result = 928 > python_version >= 912
        elif basename == 'python3.10':
            result = 944 > python_version >= 928
        elif basename == 'python3.11':
            result = 960 > python_version >= 944
        else:
            result = None
        if result is False:
            general.sysexit("The program you compiled wants to be run with: %s.\n\nNuitka is currently running with Python version '%s', which seems to not\nmatch that. Nuitka cannot guess the Python version of your source code. You\ntherefore might want to specify: '%s -m nuitka'.\n\nThat will make use the correct Python version for Nuitka.\n" % (shebang, python_version_str, binary))

def readSourceLine(source_ref):
    if False:
        print('Hello World!')
    import linecache
    return linecache.getline(filename=source_ref.getFilename(), lineno=source_ref.getLineNumber())

def writeSourceCode(filename, source_code):
    if False:
        while True:
            i = 10
    assert not os.path.isfile(filename), filename
    putTextFileContents(filename=filename, contents=source_code, encoding='latin1')

def parsePyIFile(module_name, pyi_filename):
    if False:
        i = 10
        return i + 15
    'Parse a pyi file for the given module name and extract imports made.'
    pyi_deps = OrderedSet()
    in_import = False
    in_import_part = ''
    in_quote = None
    for (line_number, line) in enumerate(getFileContentByLine(pyi_filename), start=1):
        line = line.strip()
        if in_quote:
            if line.endswith(in_quote):
                in_quote = None
            continue
        if line.startswith('"""'):
            in_quote = '"""'
            continue
        if line.startswith("'''"):
            in_quote = "'''"
            continue
        if not in_import:
            if line.startswith('import '):
                imported = line[7:]
                pyi_deps.add(imported)
            elif line.startswith('from '):
                parts = line.split(None, 3)
                assert parts[0] == 'from'
                assert parts[2] == 'import', (line, pyi_filename, line_number)
                origin_name = parts[1]
                if origin_name in ('typing', '__future__'):
                    continue
                if origin_name == '.':
                    origin_name = module_name
                else:
                    dot_count = 0
                    while origin_name.startswith('.'):
                        origin_name = origin_name[1:]
                        dot_count += 1
                    if dot_count > 0:
                        if origin_name:
                            origin_name = module_name.getRelativePackageName(level=dot_count + 1).getChildNamed(origin_name)
                        else:
                            origin_name = module_name.getRelativePackageName(level=dot_count + 1)
                if origin_name != module_name:
                    pyi_deps.add(origin_name)
                imported = parts[3]
                if imported.startswith('('):
                    if not imported.endswith(')'):
                        in_import = True
                        imported = imported[1:]
                        in_import_part = origin_name
                        assert in_import_part, 'Multiline part in file %s cannot be empty' % pyi_filename
                    else:
                        in_import = False
                        imported = imported[1:-1]
                        assert imported
                if imported == '*':
                    continue
                for name in imported.split(','):
                    if name:
                        name = name.strip()
                        pyi_deps.add(origin_name + '.' + name)
        else:
            imported = line
            if imported.endswith(')'):
                imported = imported[0:-1]
                in_import = False
            for name in imported.split(','):
                name = name.strip()
                if name:
                    pyi_deps.add(in_import_part + '.' + name)
    return pyi_deps