"""Command line tool for checking for regexes in diffs.

Filters `git diff` by path, then checks to make sure no lines matching a regex
have been added in the diff.
"""
import argparse
import dataclasses
import logging
import re
import sys
from typing import Iterable, Optional, Sequence
from xla.build_tools.lint import diff_parser

@dataclasses.dataclass
class RegexLocation:
    """Path and line where a prohibited regex was found.

  Attributes:
    path: Path of the file which has the prohibited regex.
    line_number: The number of the offending line.
    line_contents: The text of the offending line.
    matched_text: The exact string matched by the regex.
  """
    path: str
    line_number: int
    line_contents: str
    matched_text: str

def filter_hunks_by_path(hunks: Iterable[diff_parser.Hunk], *, path_regexes: list[str], path_regex_exclusions: list[str]) -> list[diff_parser.Hunk]:
    if False:
        for i in range(10):
            print('nop')
    "Filters files according to path_regexes.\n\n  If a file matches both a path_regex and a path_regex_exclusion, then\n  it will be filtered out.\n\n  Arguments:\n    hunks: A sequence of Hunk objects representing the hunks of the diff in the\n      change.\n    path_regexes: A list of regexes. Paths matching these will pass through the\n      filter. By default, every path is matched.\n    path_regex_exclusions: A list of regexes. Paths that match both a path_regex\n      and a path_regex_exclusion won't pass through the filter.\n\n  Returns:\n    A list of FileDiffs whose paths match a path_regex and don't match\n      any path_regex_exclusions.\n  "
    if not path_regexes:
        path_regexes = ['.*']
    path_regexes = [re.compile(regex) for regex in path_regexes]

    def should_include(path: str) -> bool:
        if False:
            i = 10
            return i + 15
        return any((regex.search(path) for regex in path_regexes))
    path_regex_exclusions = [re.compile(regex) for regex in path_regex_exclusions]

    def should_exclude(path: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return any((regex.search(path) for regex in path_regex_exclusions))
    return [hunk for hunk in hunks if should_include(hunk.file) and (not should_exclude(hunk.file))]

def check_diffs(hunks: Iterable[diff_parser.Hunk], *, prohibited_regex: str, suppression_regex: Optional[str]=None) -> list[RegexLocation]:
    if False:
        for i in range(10):
            print('nop')
    "Checks FileDiffs for prohibited regexes.\n\n  Arguments:\n    hunks: A sequence of Hunk objects representing the hunks of the diff.\n    prohibited_regex: The regex that isn't allowed in the diff.\n    suppression_regex: A regex used as an escape hatch to allow the prohibited\n      regex in the diff. If this is found on the same line as prohibited_regex,\n      there is no error.\n\n  Returns:\n    A list of RegexLocations where the prohibited_regex is found.\n  "
    prohibited_regex = re.compile(prohibited_regex)
    if suppression_regex is not None:
        suppression_regex = re.compile(suppression_regex)

    def should_not_suppress(line) -> bool:
        if False:
            print('Hello World!')
        if suppression_regex:
            return not suppression_regex.search(line)
        return True
    regex_locations = []
    for hunk in hunks:
        for (line_no, line) in hunk.added_lines():
            if should_not_suppress(line):
                regex_locations.extend([RegexLocation(hunk.file, line_no, line, regex_match.group()) for regex_match in prohibited_regex.finditer(line)])
    return regex_locations

def main(argv: Sequence[str]):
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Check `git diff` for prohibited regexes.')
    parser.add_argument('--path_regex', nargs='*', default=[])
    parser.add_argument('--path_regex_exclusion', nargs='*', default=[])
    parser.add_argument('--prohibited_regex', required=True)
    parser.add_argument('--suppression_regex')
    parser.add_argument('--failure_message', required=True)
    args = parser.parse_args(argv[1:])
    file_diffs = filter_hunks_by_path(diff_parser.parse_hunks(diff_parser.get_git_diff_stdout()), path_regexes=args.path_regex, path_regex_exclusions=args.path_regex_exclusion)
    regex_locations = check_diffs(file_diffs, prohibited_regex=args.prohibited_regex, suppression_regex=args.suppression_regex)
    if regex_locations:
        for loc in regex_locations:
            logging.error('Found `%s` in %s:%s', args.prohibited_regex, loc.path, loc.line_number)
            logging.error('Matched `%s` in line `%s`', loc.matched_text, loc.line_contents)
            logging.error('Failure message: %s', args.failure_message)
        sys.exit(1)
    else:
        logging.info('Prohibited regex `%s` not found in diff!', args.prohibited_regex)
        sys.exit(0)
if __name__ == '__main__':
    main(sys.argv)