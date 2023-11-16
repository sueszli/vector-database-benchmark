from __future__ import annotations
import argparse
import re
from typing import IO
from typing import Sequence
PASS = 0
FAIL = 1

class Requirement:
    UNTIL_COMPARISON = re.compile(b'={2,3}|!=|~=|>=?|<=?')
    UNTIL_SEP = re.compile(b'[^;\\s]+')

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.value: bytes | None = None
        self.comments: list[bytes] = []

    @property
    def name(self) -> bytes:
        if False:
            while True:
                i = 10
        assert self.value is not None, self.value
        name = self.value.lower()
        for egg in (b'#egg=', b'&egg='):
            if egg in self.value:
                return name.partition(egg)[-1]
        m = self.UNTIL_SEP.match(name)
        assert m is not None
        name = m.group()
        m = self.UNTIL_COMPARISON.search(name)
        if not m:
            return name
        return name[:m.start()]

    def __lt__(self, requirement: Requirement) -> bool:
        if False:
            i = 10
            return i + 15
        assert self.value is not None, self.value
        if self.value == b'\n':
            return True
        elif requirement.value == b'\n':
            return False
        else:
            return self.name < requirement.name

    def is_complete(self) -> bool:
        if False:
            print('Hello World!')
        return self.value is not None and (not self.value.rstrip(b'\r\n').endswith(b'\\'))

    def append_value(self, value: bytes) -> None:
        if False:
            while True:
                i = 10
        if self.value is not None:
            self.value += value
        else:
            self.value = value

def fix_requirements(f: IO[bytes]) -> int:
    if False:
        for i in range(10):
            print('nop')
    requirements: list[Requirement] = []
    before = list(f)
    after: list[bytes] = []
    before_string = b''.join(before)
    if before and (not before[-1].endswith(b'\n')):
        before[-1] += b'\n'
    if before_string.strip() == b'':
        return PASS
    for line in before:
        if not len(requirements) or requirements[-1].is_complete():
            requirements.append(Requirement())
        requirement = requirements[-1]
        if len(requirements) == 1 and line.strip() == b'':
            if len(requirement.comments) and requirement.comments[0].startswith(b'#'):
                requirement.value = b'\n'
            else:
                requirement.comments.append(line)
        elif line.lstrip().startswith(b'#') or line.strip() == b'':
            requirement.comments.append(line)
        else:
            requirement.append_value(line)
    if requirements[-1].value is None:
        rest = requirements.pop().comments
    else:
        rest = []
    requirements = [req for req in requirements if req.value != b'pkg-resources==0.0.0\n']
    for requirement in sorted(requirements):
        after.extend(requirement.comments)
        assert requirement.value, requirement.value
        after.append(requirement.value)
    after.extend(rest)
    after_string = b''.join(after)
    if before_string == after_string:
        return PASS
    else:
        f.seek(0)
        f.write(after_string)
        f.truncate()
        return FAIL

def main(argv: Sequence[str] | None=None) -> int:
    if False:
        return 10
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*', help='Filenames to fix')
    args = parser.parse_args(argv)
    retv = PASS
    for arg in args.filenames:
        with open(arg, 'rb+') as file_obj:
            ret_for_file = fix_requirements(file_obj)
            if ret_for_file:
                print(f'Sorting {arg}')
            retv |= ret_for_file
    return retv
if __name__ == '__main__':
    raise SystemExit(main())