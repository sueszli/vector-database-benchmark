import re
import sys
from collections import defaultdict
import pycodestyle
from flake8.formatting.default import Pylint
from flake8.style_guide import Violation
pattern_exemptions = {'package.py$': {'F403': ['^from spack.package import \\*$', '^from spack.package_defs import \\*$'], 'F811': ['^\\s*@when\\(.*\\)']}, '.py$': {'E501': ['(ssh|https?|ftp|file)\\:', '([\\\'"])[0-9a-fA-F]{32,}\\1']}}
pattern_exemptions = dict(((re.compile(file_pattern), dict(((code, [re.compile(p) for p in patterns]) for (code, patterns) in error_dict.items()))) for (file_pattern, error_dict) in pattern_exemptions.items()))

class SpackFormatter(Pylint):

    def __init__(self, options):
        if False:
            i = 10
            return i + 15
        self.spack_errors = {}
        self.error_seen = False
        super().__init__(options)

    def after_init(self) -> None:
        if False:
            print('Hello World!')
        'Overriding to keep format string from being unset in Default'
        pass

    def beginning(self, filename):
        if False:
            for i in range(10):
                print('nop')
        self.filename = filename
        self.file_lines = None
        self.spack_errors = defaultdict(list)
        for (file_pattern, errors) in pattern_exemptions.items():
            if file_pattern.search(filename):
                for (code, pat_arr) in errors.items():
                    self.spack_errors[code].extend(pat_arr)

    def handle(self, error: Violation) -> None:
        if False:
            while True:
                i = 10
        'Handle an error reported by Flake8.\n\n        This defaults to calling :meth:`format`, :meth:`show_source`, and\n        then :meth:`write`. This version implements the pattern-based ignore\n        behavior from `spack flake8` as a native flake8 plugin.\n\n        :param error:\n            This will be an instance of\n            :class:`~flake8.style_guide.Violation`.\n        '
        pats = self.spack_errors.get(error.code, None)
        if pats is not None and any((pat.search(error.physical_line) for pat in pats)):
            return
        if self.spack_errors.get('F811', False) and error.code == 'F811' and (error.line_number > 1):
            if self.file_lines is None:
                if self.filename in {'stdin', '-', '(none)', None}:
                    self.file_lines = pycodestyle.stdin_get_value().splitlines(True)
                else:
                    self.file_lines = pycodestyle.readlines(self.filename)
            for pat in self.spack_errors['F811']:
                if pat.search(self.file_lines[error.line_number - 2]):
                    return
        self.error_seen = True
        line = self.format(error)
        source = self.show_source(error)
        self.write(line, source)

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        'Override stop to check whether any errors we consider to be errors\n        were reported.\n\n        This is a hack, but it makes flake8 behave the desired way.\n        '
        if not self.error_seen:
            sys.exit(0)