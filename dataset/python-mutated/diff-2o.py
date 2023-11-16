"""
Produces an approximate second-order diff of the current commit and the linked
diff in Phabricator. This is useful for seeing roughly what you changed since
you last uploaded a commit.

This is often much clearer than doing something like:
    git diff HEAD <commit uploaded to Phabricator>
as it only looks at what changed between the two commit diffs. This excludes
changes which are the result of the local commit being rebased, but have
nothing to do with what you actually changed.

The output is only approximate beacuse there's generally not information to
fully reconstruct the changes by just looking at two diffs. However, most of the
time it gives a pretty good hint to a human what changed.
"""
import difflib
import re
import shlex
import subprocess
from typing import Dict, List, Union

def _run(cmd: Union[str, List[str]]) -> List[str]:
    if False:
        while True:
            i = 10
    "Run a 'cmd', returning stdout as a list of strings."
    cmd_list = shlex.split(cmd) if type(cmd) == str else cmd
    result = subprocess.run(cmd_list, capture_output=True)
    return result.stdout.decode('utf-8').split('\n')

def _split_diff_to_context_free_file_diffs(diff: List[str], src: str) -> Dict[str, List[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Take a noisy diff-like input covering many files and return a purified\n    map of file name -> hunk changes. Strips out all line info, metadata, and\n    context along the way. E.g. for input:\n        Summary\n        diff --git a/dir/fileXYZ b/dir/fileXYZ\n        --- a/dir/fileXYZ\n        +++ b/dir/fileXYZ\n        @@ -1,3 +1,3 @@ inferred function context\n         context\n        -badline\n        +goodline\n         context\n     Produce:\n        {"fileXYZ": [\n            "@@",\n            "-badline",\n            "+goodline",\n        ]}\n    '
    res: Dict[str, List[str]] = {}
    current_diff_file_lines: List[str] = []
    line_n = 0
    while line_n < len(diff):
        line = diff[line_n]
        m = re.match('^--- a/(.*)$', line)
        if m:
            current_diff_file_lines = []
            filename = m.group(1)
            res[filename] = current_diff_file_lines
            if line_n > len(diff) - 1:
                raise Exception(f'{src}:{line_n} - missing +++ after ---')
            line_n += 1
            line = diff[line_n]
            if not re.match(f'^\\+\\+\\+ b/{filename}', line):
                raise Exception(f'{src}:{line_n} - invalid +++ line after ---')
        elif line:
            if line[:2] == '@@':
                current_diff_file_lines.append('@@')
            elif line[0] in ['+', '-']:
                current_diff_file_lines.append(diff[line_n])
        line_n += 1
    return res

def _do_diff_2o(diff_a: str, src_a: str, diff_b: str, src_b: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    file_diffs_a = _split_diff_to_context_free_file_diffs(diff_a, src_a)
    file_diffs_b = _split_diff_to_context_free_file_diffs(diff_b, src_b)
    modifiedfiles = set(file_diffs_a) & set(file_diffs_b)
    for f in modifiedfiles:
        diff_lines = difflib.unified_diff(file_diffs_a[f], file_diffs_b[f], n=0)
        diff_lines = list(diff_lines)[2:]
        i = 0
        changelines = []
        while i < len(diff_lines):
            line = diff_lines[i]
            i += 1
            if line[:2] == '++':
                changelines.append('+' + line[2:])
            elif line[:2] == '+-':
                changelines.append('-' + line[2:])
            elif line[:2] == '-+':
                changelines.append('-' + line[2:])
            elif line[:2] == '--':
                changelines.append('+' + line[2:])
            elif line[:2] == '@@' or line[1:3] == '@@':
                if len(changelines) < 1 or changelines[-1] != '...\n':
                    changelines.append('...\n')
            else:
                changelines.append(line)
        if len(changelines):
            print(f'Changed: {f}')
            for line in changelines:
                print(f'| {line.strip()}')
    wholefilechanges = set(file_diffs_a) ^ set(file_diffs_b)
    for f in wholefilechanges:
        print(f'Added/removed: {f}')

def main() -> None:
    if False:
        i = 10
        return i + 15
    git_full_diff = _run('git show')
    diff_n = None
    for line in git_full_diff:
        m = re.match('.*https://.*/(D\\d+)$', line)
        if m:
            diff_n = m.group(1)
            break
    if not diff_n:
        raise Exception('Could not find Phabricator diff from Git commit')
    phab_full_diff = _run(f'jf export --diff {diff_n}')
    _do_diff_2o(phab_full_diff, 'phab_full_diff', git_full_diff, 'git')
if __name__ == '__main__':
    main()