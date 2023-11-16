import sys, os, io, yaml, re, functools
open = functools.partial(__builtins__.open, encoding=os.environ.get('SOURCE_ENCODING', 'utf8'))
TAB = '\t'
EOL = '\n'
DIGITS = re.compile('[0-9][0-9]?')

def cached(path, cache={}):
    if False:
        print('Hello World!')
    if path not in cache:
        with open(path) as infile:
            cache[path] = infile.read().rstrip()
    return cache[path]

class DummyFile:

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        pass

    def write(self, text):
        if False:
            i = 10
            return i + 15
        pass

    def close(self):
        if False:
            return 10
        pass

class AutoDict(dict):

    def __init__(self, T):
        if False:
            for i in range(10):
                print('nop')
        self.T = T

    def __missing__(self, key):
        if False:
            while True:
                i = 10
        self[key] = self.T()
        return self[key]

class Snipper:

    def __init__(self, snippetdir):
        if False:
            i = 10
            return i + 15
        self.dir = snippetdir
        self.source = {}
        self.count = 0
        self.errors = 0
        self.issues = AutoDict(set)
        self.index = AutoDict(list)
        self.log = io.StringIO()

    def __enter__(self):
        if False:
            print('Hello World!')
        global print
        print = functools.partial(__builtins__.print, file=self.log)
        return self

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        global print
        print = __builtins__.print

    def __call__(self, path, markers):
        if False:
            i = 10
            return i + 15
        print(path)
        self.started = set()
        self.duplicates = set()
        tag = re.compile(f" *({'|'.join(markers)}) ?snippet-")
        self.files = {}
        self.dedent = {}
        self.path = path
        self.markers = markers
        try:
            with open(path) as infile:
                self.text = infile.read().rstrip()
        except IOError as ex:
            print('ERROR reading file', ex)
            self.errors += 1
            return
        if TAB in self.text and 'snippet-start' in self.text:
            print('    WARNING tab(s) found in %s may cause formatting problems in docs' % path)
        for (self.i, self.line) in enumerate(self.text.splitlines(keepends=False), start=1):
            line = self.line
            if tag.match(line):
                self.directive = line.split('snippet-')[1].split(':')[0].rstrip()
                self.arg = line.split('[')[1].split(']')[0].rstrip()
                func = getattr(self, self.directive.lstrip('_'), None)
                if func and callable(func):
                    func(self.arg)
                else:
                    print('    ERROR invalid directive snippet-%s at %s in %s' % (self.directive, self.i, self.path))
                    self.errors += 1
                    self.issues[path].add('invalid directive snippet-%s' % self.directive)
            else:
                for (snip, file) in self.files.items():
                    dedent = self.dedent[snip]
                    if dedent and line[:dedent].strip():
                        print('    ERROR unable to dedent %s space(s) ' % dedent + 'in snippet %s at line %s in %s ' % self._where + f'(only indented {len(line) - len(line.lstrip())} spaces)')
                        self.errors += 1
                    file.write(line[dedent:].rstrip() + EOL)
        for (snip, file) in self.files.items():
            print('    ERROR snippet-end tag for %s missing in %s, extracted to end of file' % (snip, path))
            file.close()
            self.issues[path].add('snippet-end tag for %s missing' % snip)
            self.errors += 1

    def start(self, arg):
        if False:
            print('Hello World!')
        path = os.path.join(self.dir, f'{arg}.txt')
        indicator = 'EXTRACT'
        opener = open
        printer = print
        if arg in self.files:
            printer = lambda *a: print('    ERROR snippet %s already open at line %s in %s' % self._where)
            self.issues[self.path].add('snippet %s opened multiple times')
            self.errors += 1
        elif os.path.isfile(path):
            if self.path != self.source[arg] and self.path.rpartition('/')[2] == self.source[arg].rpartition('/')[2] and (self.text == cached(self.source[arg])):
                printer = lambda *a: print('WARNING redundant snippet %s at line %s in %s' % self._where)
                self.duplicates.add(arg)
            else:
                printer = lambda *a: print('    ERROR duplicate snippet %s at line %s in %s' % self._where, '(also in %s)' % self.source[arg])
                pfxlen = len(os.path.commonprefix([self.path, self.source[arg]]))
                path1 = self.source[arg][pfxlen:]
                if '/' not in path1:
                    path1 = self.source[arg]
                path2 = self.path[pfxlen:]
                if '/' not in path2:
                    path2 = self.path
                self.issues[self.path].add('%s also declared in %s' % (arg, path1))
                self.issues[self.source[arg]].add('%s also declared in %s' % (arg, path2))
                self.errors += 1
            opener = DummyFile
        else:
            self.count += 1
        self.dedent[arg] = int(DIGITS.search(self.line.rpartition(']')[2] + ' 0').group(0))
        self.files[arg] = opener(path, 'w')
        self.index[arg].append(self.path)
        self.started.add(arg)
        if arg not in self.source:
            self.source[arg] = self.path
        printer('   ', indicator, arg)

    def append(self, arg):
        if False:
            i = 10
            return i + 15
        if arg in self.files:
            print('    ERROR snippet %s already open at line %s in %s' % self._where)
            self.issues[self, path].add('snippet %s opened multiple times' % arg)
            self.errors += 1
            return
        if arg not in self.started:
            print('    ERROR snippet file %s not found at line %s in %s' % self._where)
            self.issues[self.path].add("snippet %s doesn't exist" % arg)
            self.errors += 1
            return
        self.files[arg] = DummyFile() if arg in self.duplicates else open(os.path.join(self.dir, arg) + '.txt', 'a')
        print('    APPEND', arg)

    def end(self, arg):
        if False:
            return 10
        if arg in self.files:
            self.files[arg].close()
            del self.files[arg]
        else:
            print('    ERROR snippet file %s not open at %s in %s' % self._where)
            self.issues[self.path].add('snippet-end tag for %s which is not open' % arg)
            self.errors += 1

    def echo(self, arg):
        if False:
            i = 10
            return i + 15
        arg = arg.rstrip() + EOL
        if self.files:
            for file in self.files.values():
                file.write(arg)
        else:
            print("    ERROR echo '%s' outside snippet at %s in %s" % self._where)
            self.issues[self.path].add('echo outside snippet')
            self.errors += 1

    def _nop(self, arg):
        if False:
            return 10
        return
    service = comment = keyword = sourceauthor = sourcedate = sourcedescription = sourcetype = sourcesyntax = _nop

    @property
    def _where(self):
        if False:
            print('Hello World!')
        return (self.arg, self.i, self.path)

def err_exit(msg):
    if False:
        while True:
            i = 10
    print('ERROR', msg)
    sys.exit(1)
if __name__ == '__main__':
    stdin_lines = []
    if not sys.stdin.isatty():
        stdin_lines = sys.stdin.readlines()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        snippetdir = sys.argv[1]
    else:
        err_exit('snippet output directory not passed or does not exist')
    if len(sys.argv) > 2:
        commentfile = sys.argv[2]
    else:
        commentfile = 'snippet-extensions.yml'
    reports = os.environ.get('REPORTS', 'log issues index').lower().split()
    if '/' not in commentfile and '\\' not in commentfile:
        commentfile = os.path.join(os.path.dirname(__file__), commentfile)
    if not os.path.isfile(commentfile):
        err_exit('source file extension map %s not found' % commentfile)
    with open(commentfile) as comments:
        MAP_EXT_MARKER = yaml.safe_load(comments)
        if not isinstance(MAP_EXT_MARKER, dict):
            err_exit('source map is not a key-value store (dictionary)')
        for (k, v) in MAP_EXT_MARKER.items():
            if isinstance(k, str) and isinstance(v, str):
                MAP_EXT_MARKER[k] = v.split()
            else:
                err_exit('key, value must both be strings; got %s, %s (%s, %s)' % (k, v, type(k).__name__, type(v).__name__))
    print('==== extracting snippets in source files', ' '.join((ex for ex in MAP_EXT_MARKER if ex and MAP_EXT_MARKER[ex])), '\n')
    print('reports:', ' '.join(reports).upper(), end='\n\n')
    with Snipper(snippetdir) as snipper:
        seen = processed = 0
        for path in sorted(stdin_lines):
            path = path.strip()
            if not path:
                continue
            if not (path.startswith(('./', '/', '\\')) or (path[0].isalpha() and path[1] == ':')):
                path = './' + path
            if '/.' in path or '\\.' in path:
                continue
            seen += 1
            ext = next((ext for ext in MAP_EXT_MARKER if path.replace('\\', '/').endswith(ext)), None)
            markers = MAP_EXT_MARKER.get(ext, ())
            if markers:
                snipper(path, markers)
                processed += 1
    if 'issues' in reports:
        if snipper.issues:
            print('====', len(snipper.issues), 'file(s) with issues:', end='\n\n')
            for (issue, details) in sorted(snipper.issues.items(), key=lambda item: -len(item[1])):
                print(issue, end='\n     ')
                print(*sorted(details), sep='\n     ', end='\n\n')
        else:
            print('---- no issues found\n')
    if 'index' in reports:
        if snipper.index:
            print('====', len(snipper.index), 'snippet(s) extracted from', processed, 'files:', end='\n\n')
            for (snippet, files) in sorted(snipper.index.items(), key=lambda item: -len(item[1])):
                print(snippet, 'declared in:', end='\n     ')
                print(*sorted(files), sep='\n     ', end='\n\n')
        else:
            print('--- no snippets were extracted\n')
    if 'log' in reports:
        print('==== Complete processing log\n')
        if processed:
            print(snipper.log.getvalue(), end='\n\n')
        else:
            print('No files were processed\n')
    print('====', snipper.count, 'snippet(s) extracted from', processed, 'source file(s) processed of', seen, 'candidate(s) with', snipper.errors, 'error(s) in', len(snipper.issues), 'file(s)\n')
    sys.exit(snipper.errors > 0)