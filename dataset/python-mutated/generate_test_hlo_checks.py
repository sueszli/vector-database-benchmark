"""Replace HLO instruction names with FileCheck variable captures.c.

Makes FileCheck-based tests on HLO more resilient to instruction name changes.
"""
import re
import shutil
import sys
import tempfile
from typing import Dict
ESCAPE_FILECHECK_VARNAME = re.compile('[^a-zA-Z0-9_]')

class FileCheckVarReplacer:
    """Replacer class for replacing HLO instructions by FileCheck captures."""
    _counter: int
    _replacement_cache: Dict[str, str]
    _check_instruction_matcher: re.Pattern = re.compile('^[^:]*CHECK[^:]*:.*=')
    _instr_name_matcher: re.Pattern = re.compile('%[\\w-]+(\\.\\d+)?')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._counter = -1
        self._replacement_cache = {}

    def replace_instruction_names_for_line(self, line: str) -> str:
        if False:
            return 10
        'Replaces all HLO instruction names by captured FileCheck variables.\n\n    Works only for instruction definitions preceded by "CHECK-XXX: " directives.\n\n    Args:\n      line: One of test lines.\n\n    Returns:\n      A line with replacements applied.\n    '
        if not self._check_instruction_matcher.match(line):
            self._counter = -1
            self._replacement_cache = {}
            return line
        return re.sub(self._instr_name_matcher, self._replacer, line)

    def _replacer(self, m: re.Match) -> str:
        if False:
            print('Hello World!')
        instr_name = m.group(0)
        if instr_name in self._replacement_cache:
            return self._replacement_cache[instr_name]
        replacement_instr = self._generate_unique_varname(instr_name)
        self._replacement_cache[instr_name] = f'[[{replacement_instr}]]'
        return ''.join([f'[[{replacement_instr}:', '%[^ ]+', ']]'])

    def _generate_unique_varname(self, instr_name: str) -> str:
        if False:
            i = 10
            return i + 15
        self._counter += 1
        normalized_instr_name = ESCAPE_FILECHECK_VARNAME.sub('_', instr_name.replace('%', ''))
        return f'{normalized_instr_name}_{self._counter}'

def replace_instruction_names(t: str) -> str:
    if False:
        print('Hello World!')
    'Replaces all HLO instruction names by captured FileCheck variables.\n\n  Args:\n    t: Test text to replace\n\n  Returns:\n    Test with replacements applied.\n  '
    f = FileCheckVarReplacer()
    out = []
    for line in t.split('\n'):
        out.append(f.replace_instruction_names_for_line(line))
    return '\n'.join(out)

def main() -> None:
    if False:
        return 10
    argv = sys.argv
    if len(argv) != 2:
        raise Exception('Expecting exactly one filename argument (or -)')
    r = FileCheckVarReplacer()
    input_filename = argv[1]
    if input_filename == '-':
        for line in sys.stdin:
            sys.stdout.write(r.replace_instruction_names_for_line(line))
        return 0
    with open(input_filename) as f:
        (fd, fname) = tempfile.mkstemp()
        with open(fd, 'w') as out_f:
            for line in f:
                out_f.write(r.replace_instruction_names_for_line(line))
    shutil.move(fname, input_filename)
if __name__ == '__main__':
    main()