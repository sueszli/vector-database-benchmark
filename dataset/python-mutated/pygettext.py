import os
import imp
import sys
import glob
import token
import tokenize
__version__ = '1.5'
default_keywords = ['_']
DEFAULTKEYWORDS = ', '.join(default_keywords)
EMPTYSTRING = ''
pot_header = '\nmsgid ""\nmsgstr ""\n"Content-Type: text/plain; charset=utf-8\\n"\n"Content-Transfer-Encoding: utf-8\\n"\n'

def usage(code, msg=''):
    if False:
        print('Hello World!')
    print(__doc__ % globals(), file=sys.stderr)
    if msg:
        print(msg, file=sys.stderr)
    sys.exit(code)
escapes = []

def make_escapes(pass_iso8859):
    if False:
        return 10
    global escapes
    if pass_iso8859:
        mod = 128
    else:
        mod = 256
    for i in range(256):
        if 32 <= i % mod <= 126:
            escapes.append(chr(i))
        else:
            escapes.append('\\%03o' % i)
    escapes[ord('\\')] = '\\\\'
    escapes[ord('\t')] = '\\t'
    escapes[ord('\r')] = '\\r'
    escapes[ord('\n')] = '\\n'
    escapes[ord('"')] = '\\"'

def escape(s):
    if False:
        i = 10
        return i + 15
    global escapes
    s = list(s)
    for i in range(len(s)):
        s[i] = escapes[ord(s[i])]
    return EMPTYSTRING.join(s)

def safe_eval(s):
    if False:
        print('Hello World!')
    return eval(s, {'__builtins__': {}}, {})

def normalize(s):
    if False:
        return 10
    lines = s.split('\n')
    if len(lines) == 1:
        s = '"' + escape(s) + '"'
    else:
        if not lines[-1]:
            del lines[-1]
            lines[-1] = lines[-1] + '\n'
        for i in range(len(lines)):
            lines[i] = escape(lines[i])
        lineterm = '\\n"\n"'
        s = '""\n"' + lineterm.join(lines) + '"'
    return s

def containsAny(str, set):
    if False:
        while True:
            i = 10
    "Check whether 'str' contains ANY of the chars in 'set'"
    return 1 in [c in str for c in set]

def _visit_pyfiles(list, dirname, names):
    if False:
        print('Hello World!')
    'Helper for getFilesForName().'
    if '_py_ext' not in globals():
        global _py_ext
        _py_ext = [triple[0] for triple in imp.get_suffixes() if triple[2] == imp.PY_SOURCE][0]
    if 'CVS' in names:
        names.remove('CVS')
    list.extend([os.path.join(dirname, file) for file in names if os.path.splitext(file)[1] == _py_ext])

def _get_modpkg_path(dotted_name, pathlist=None):
    if False:
        print('Hello World!')
    'Get the filesystem path for a module or a package.\n\n    Return the file system path to a file for a module, and to a directory for\n    a package. Return None if the name is not found, or is a builtin or\n    extension module.\n    '
    parts = dotted_name.split('.', 1)
    if len(parts) > 1:
        try:
            (file, pathname, description) = imp.find_module(parts[0], pathlist)
            if file:
                file.close()
        except ImportError:
            return None
        if description[2] == imp.PKG_DIRECTORY:
            pathname = _get_modpkg_path(parts[1], [pathname])
        else:
            pathname = None
    else:
        try:
            (file, pathname, description) = imp.find_module(dotted_name, pathlist)
            if file:
                file.close()
            if description[2] not in [imp.PY_SOURCE, imp.PKG_DIRECTORY]:
                pathname = None
        except ImportError:
            pathname = None
    return pathname

def getFilesForName(name):
    if False:
        for i in range(10):
            print('nop')
    'Get a list of module files for a filename, a module or package name,\n    or a directory.\n    '
    if not os.path.exists(name):
        if containsAny(name, '*?[]'):
            files = glob.glob(name)
            file_list = []
            for file in files:
                file_list.extend(getFilesForName(file))
            return file_list
        name = _get_modpkg_path(name)
        if not name:
            return []
    if os.path.isdir(name):
        file_list = []
        os.walk(name, _visit_pyfiles, file_list)
        return file_list
    elif os.path.exists(name):
        return [name]
    return []

class TokenEater:

    def __init__(self, options):
        if False:
            while True:
                i = 10
        self.__options = options
        self.__messages = {}
        self.__state = self.__waiting
        self.__data = []
        self.__lineno = -1
        self.__freshmodule = 1
        self.__curfile = None

    def __call__(self, ttype, tstring, stup, etup, line):
        if False:
            for i in range(10):
                print('nop')
        self.__state(ttype, tstring, stup[0])

    def __waiting(self, ttype, tstring, lineno):
        if False:
            i = 10
            return i + 15
        opts = self.__options
        if opts.docstrings and (not opts.nodocstrings.get(self.__curfile)):
            if self.__freshmodule:
                if ttype == tokenize.STRING:
                    self.__addentry(safe_eval(tstring), lineno, isdocstring=1)
                    self.__freshmodule = 0
                elif ttype not in (tokenize.COMMENT, tokenize.NL):
                    self.__freshmodule = 0
                return
            if ttype == tokenize.NAME and tstring in ('class', 'def'):
                self.__state = self.__suiteseen
                return
        if ttype == tokenize.NAME and tstring in opts.keywords:
            self.__state = self.__keywordseen

    def __suiteseen(self, ttype, tstring, lineno):
        if False:
            while True:
                i = 10
        if ttype == tokenize.OP and tstring == ':':
            self.__state = self.__suitedocstring

    def __suitedocstring(self, ttype, tstring, lineno):
        if False:
            while True:
                i = 10
        if ttype == tokenize.STRING:
            self.__addentry(safe_eval(tstring), lineno, isdocstring=1)
            self.__state = self.__waiting
        elif ttype not in (tokenize.NEWLINE, tokenize.INDENT, tokenize.COMMENT):
            self.__state = self.__waiting

    def __keywordseen(self, ttype, tstring, lineno):
        if False:
            print('Hello World!')
        if ttype == tokenize.OP and tstring == '(':
            self.__data = []
            self.__lineno = lineno
            self.__state = self.__openseen
        else:
            self.__state = self.__waiting

    def __openseen(self, ttype, tstring, lineno):
        if False:
            return 10
        if ttype == tokenize.OP and tstring == ')':
            if self.__data:
                self.__addentry(EMPTYSTRING.join(self.__data))
            self.__state = self.__waiting
        elif ttype == tokenize.STRING:
            self.__data.append(safe_eval(tstring))
        elif ttype not in [tokenize.COMMENT, token.INDENT, token.DEDENT, token.NEWLINE, tokenize.NL]:
            print('*** %(file)s:%(lineno)s: Seen unexpected token "%(token)s"' % {'token': tstring, 'file': self.__curfile, 'lineno': self.__lineno}, file=sys.stderr)
            self.__state = self.__waiting

    def __addentry(self, msg, lineno=None, isdocstring=0):
        if False:
            while True:
                i = 10
        if lineno is None:
            lineno = self.__lineno
        if msg not in self.__options.toexclude:
            entry = (self.__curfile, lineno)
            self.__messages.setdefault(msg, {})[entry] = isdocstring

    def set_filename(self, filename):
        if False:
            return 10
        self.__curfile = filename
        self.__freshmodule = 1

    def write(self, fp):
        if False:
            for i in range(10):
                print('nop')
        options = self.__options
        print(pot_header, file=fp)
        reverse = {}
        for (k, v) in self.__messages.items():
            keys = sorted(v.keys())
            reverse.setdefault(tuple(keys), []).append((k, v))
        rkeys = sorted(reverse.keys())
        for rkey in rkeys:
            rentries = reverse[rkey]
            rentries.sort()
            for (k, v) in rentries:
                isdocstring = any(v.values())
                v = sorted(v.keys())
                if not options.writelocations:
                    pass
                elif options.locationstyle == options.SOLARIS:
                    for (filename, lineno) in v:
                        d = {'filename': filename, 'lineno': lineno}
                        print('# File: %(filename)s, line: %(lineno)d' % d, file=fp)
                elif options.locationstyle == options.GNU:
                    locline = '#:'
                    for (filename, lineno) in v:
                        d = {'filename': filename, 'lineno': lineno}
                        s = ' %(filename)s:%(lineno)d' % d
                        if len(locline) + len(s) <= options.width:
                            locline = locline + s
                        else:
                            print(locline, file=fp)
                            locline = '#:' + s
                    if len(locline) > 2:
                        print(locline, file=fp)
                if isdocstring:
                    print('#, docstring', file=fp)
                print('msgid', normalize(k), file=fp)
                print('msgstr ""\n', file=fp)

def main(source_files, outpath, keywords=None):
    if False:
        for i in range(10):
            print('nop')
    global default_keywords

    class Options:
        GNU = 1
        SOLARIS = 2
        extractall = 0
        escape = 0
        keywords = []
        outfile = 'messages.pot'
        writelocations = 1
        locationstyle = GNU
        verbose = 0
        width = 78
        excludefilename = ''
        docstrings = 0
        nodocstrings = {}
    options = Options()
    options.outfile = outpath
    if keywords:
        options.keywords = keywords
    make_escapes(options.escape)
    options.keywords.extend(default_keywords)
    if options.excludefilename:
        try:
            fp = open(options.excludefilename, encoding='utf-8')
            options.toexclude = fp.readlines()
            fp.close()
        except OSError:
            print("Can't read --exclude-file: %s" % options.excludefilename, file=sys.stderr)
            sys.exit(1)
    else:
        options.toexclude = []
    eater = TokenEater(options)
    for filename in source_files:
        if options.verbose:
            print('Working on %s' % filename)
        fp = open(filename, encoding='utf-8')
        closep = 1
        try:
            eater.set_filename(filename)
            try:
                tokens = tokenize.generate_tokens(fp.readline)
                for _token in tokens:
                    eater(*_token)
            except tokenize.TokenError as e:
                print('%s: %s, line %d, column %d' % (e.args[0], filename, e.args[1][0], e.args[1][1]), file=sys.stderr)
        finally:
            if closep:
                fp.close()
    fp = open(options.outfile, 'w', encoding='utf-8')
    closep = 1
    try:
        eater.write(fp)
    finally:
        if closep:
            fp.close()