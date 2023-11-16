import textwrap
from robot.errors import DataError
from robot.utils import MultiMatcher, console_encode

class ConsoleViewer:

    def __init__(self, libdoc):
        if False:
            i = 10
            return i + 15
        self._libdoc = libdoc
        self._keywords = KeywordMatcher(libdoc)

    @classmethod
    def handles(cls, command):
        if False:
            while True:
                i = 10
        return command.lower() in ['list', 'show', 'version']

    @classmethod
    def validate_command(cls, command, args):
        if False:
            i = 10
            return i + 15
        if not cls.handles(command):
            raise DataError("Unknown command '%s'." % command)
        if command.lower() == 'version' and args:
            raise DataError("Command 'version' does not take arguments.")

    def view(self, command, *args):
        if False:
            i = 10
            return i + 15
        self.validate_command(command, args)
        getattr(self, command.lower())(*args)

    def list(self, *patterns):
        if False:
            for i in range(10):
                print('nop')
        for kw in self._keywords.search(('*%s*' % p for p in patterns)):
            self._console(kw.name)

    def show(self, *names):
        if False:
            print('Hello World!')
        if MultiMatcher(names, match_if_no_patterns=True).match('intro'):
            self._show_intro(self._libdoc)
            if self._libdoc.inits:
                self._show_inits(self._libdoc)
        for kw in self._keywords.search(names):
            self._show_keyword(kw)

    def version(self):
        if False:
            print('Hello World!')
        self._console(self._libdoc.version or 'N/A')

    def _console(self, msg):
        if False:
            print('Hello World!')
        print(console_encode(msg))

    def _show_intro(self, lib):
        if False:
            while True:
                i = 10
        self._header(lib.name, underline='=')
        self._data([('Version', lib.version), ('Scope', lib.scope if lib.type == 'LIBRARY' else None)])
        self._doc(lib.doc)

    def _show_inits(self, lib):
        if False:
            i = 10
            return i + 15
        self._header('Importing', underline='-')
        for init in lib.inits:
            self._show_keyword(init, show_name=False)

    def _show_keyword(self, kw, show_name=True):
        if False:
            while True:
                i = 10
        if show_name:
            self._header(kw.name, underline='-')
        self._data([('Arguments', '[%s]' % str(kw.args))])
        self._doc(kw.doc)

    def _header(self, name, underline):
        if False:
            for i in range(10):
                print('nop')
        self._console('%s\n%s' % (name, underline * len(name)))

    def _data(self, items):
        if False:
            while True:
                i = 10
        ljust = max((len(name) for (name, _) in items)) + 3
        for (name, value) in items:
            if value:
                text = '%s%s' % ((name + ':').ljust(ljust), value)
                self._console(self._wrap(text, subsequent_indent=' ' * ljust))

    def _doc(self, doc):
        if False:
            for i in range(10):
                print('nop')
        self._console('')
        for line in doc.splitlines():
            self._console(self._wrap(line))
        if doc:
            self._console('')

    def _wrap(self, text, width=78, **config):
        if False:
            while True:
                i = 10
        return '\n'.join(textwrap.wrap(text, width=width, **config))

class KeywordMatcher:

    def __init__(self, libdoc):
        if False:
            i = 10
            return i + 15
        self._keywords = libdoc.keywords

    def search(self, patterns):
        if False:
            for i in range(10):
                print('nop')
        matcher = MultiMatcher(patterns, match_if_no_patterns=True)
        for kw in self._keywords:
            if matcher.match(kw.name):
                yield kw