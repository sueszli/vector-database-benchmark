"""Parses diffs into hunks."""
import dataclasses
import re
import subprocess
from typing import Generator, Iterable, TypeVar
_T = TypeVar('_T')

@dataclasses.dataclass(frozen=True)
class Hunk:
    """Represents a hunk of a diff."""
    file: str
    start: int
    length: int
    lines: list[str]

    def added_lines(self) -> Generator[tuple[int, str], None, None]:
        if False:
            print('Hello World!')
        current_line_no = self.start
        for line in self.lines:
            if line.startswith('+'):
                yield (current_line_no, line[1:])
                current_line_no += 1
            elif line.startswith('-'):
                continue
            else:
                current_line_no += 1

def batch(iterable: Iterable[_T], n: int) -> Generator[tuple[_T, ...], None, None]:
    if False:
        i = 10
        return i + 15
    'Splits an iterable into chunks of size n.\n\n  TODO(ddunleavy): once python 3.12 is available, use itertools.batch.\n\n  Arguments:\n    iterable: the iterable to batch.\n    n: the number of elements in each batch.\n\n  Yields:\n    A tuple of length n of the type that the iterable produces.\n  '
    iterator = iter(iterable)
    while True:
        try:
            yield tuple([next(iterator) for _ in range(n)])
        except StopIteration:
            return

def parse_hunks(diff: str) -> list[Hunk]:
    if False:
        print('Hello World!')
    'Parses a diff into hunks.\n\n  Arguments:\n    diff: The raw output of git diff.\n\n  Returns:\n    A list of Hunks.\n  '
    diff_pattern = 'diff --git a/.* b/(.*)\\n(?:\\w+ file mode \\d+\\n)?index .*\\n--- .*\\n\\+\\+\\+ .*\\n'
    hunk_header_pattern = '@@ -\\d+,\\d+ \\+(\\d+),(\\d+) @@.*\\n'
    raw_per_file_hunks = re.split(diff_pattern, diff)[1:]
    parsed_hunks = []
    for (file, raw_hunks) in batch(raw_per_file_hunks, 2):
        hunks = re.split(hunk_header_pattern, raw_hunks, re.MULTILINE)[1:]
        for (start, length, body) in batch(hunks, 3):
            lines = body.split('\n')
            lines = lines if lines[-1] else lines[:-1]
            parsed_hunks.append(Hunk(file, int(start), int(length), lines))
    return parsed_hunks

def get_git_diff_stdout() -> str:
    if False:
        print('Hello World!')
    'Run git diff with appropriate arguments and capture stdout as a str.'
    proc = subprocess.run(['git', 'diff', 'origin/main', 'HEAD'], capture_output=True, check=True, text=True)
    return proc.stdout