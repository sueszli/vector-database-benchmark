from __future__ import annotations
import os
import pathlib
import re
import sys

def main():
    if False:
        for i in range(10):
            print('nop')
    constraints_path = 'test/lib/ansible_test/_data/requirements/constraints.txt'
    requirements = {}
    for path in sys.argv[1:] or sys.stdin.read().splitlines():
        if path == 'test/lib/ansible_test/_data/requirements/ansible.txt':
            continue
        with open(path, 'r') as path_fd:
            requirements[path] = parse_requirements(path_fd.read().splitlines())
        if path == 'test/lib/ansible_test/_data/requirements/ansible-test.txt':
            check_ansible_test(path, requirements.pop(path))
            continue
    frozen_sanity = {}
    non_sanity_requirements = set()
    for (path, requirements) in requirements.items():
        filename = os.path.basename(path)
        is_sanity = filename.startswith('sanity.') or filename.endswith('.requirements.txt')
        is_constraints = path == constraints_path
        for (lineno, line, requirement) in requirements:
            if not requirement:
                print('%s:%d:%d: cannot parse requirement: %s' % (path, lineno, 1, line))
                continue
            name = requirement.group('name').lower()
            raw_constraints = requirement.group('constraints')
            constraints = raw_constraints.strip()
            comment = requirement.group('comment')
            is_pinned = re.search('^ *== *[0-9.]+(\\.post[0-9]+)?$', constraints)
            if is_sanity:
                sanity = frozen_sanity.setdefault(name, [])
                sanity.append((path, lineno, line, requirement))
            elif not is_constraints:
                non_sanity_requirements.add(name)
            if is_sanity:
                if not is_pinned:
                    print('%s:%d:%d: sanity test requirement (%s%s) must be frozen (use `==`)' % (path, lineno, 1, name, raw_constraints))
                continue
            if constraints and (not is_constraints):
                allow_constraints = 'sanity_ok' in comment
                if not allow_constraints:
                    print('%s:%d:%d: put the constraint (%s%s) in `%s`' % (path, lineno, 1, name, raw_constraints, constraints_path))

def check_ansible_test(path: str, requirements: list[tuple[int, str, re.Match]]) -> None:
    if False:
        while True:
            i = 10
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.joinpath('lib')))
    from ansible_test._internal.coverage_util import COVERAGE_VERSIONS
    from ansible_test._internal.util import version_to_str
    expected_lines = set((f"coverage == {item.coverage_version} ; python_version >= '{version_to_str(item.min_python)}' and python_version <= '{version_to_str(item.max_python)}'" for item in COVERAGE_VERSIONS))
    for (idx, requirement) in enumerate(requirements):
        (lineno, line, match) = requirement
        if line in expected_lines:
            expected_lines.remove(line)
            continue
        print('%s:%d:%d: unexpected line: %s' % (path, lineno, 1, line))
    for expected_line in sorted(expected_lines):
        print('%s:%d:%d: missing line: %s' % (path, requirements[-1][0] + 1, 1, expected_line))

def parse_requirements(lines):
    if False:
        while True:
            i = 10
    pattern = re.compile('^(?P<name>[A-Z0-9][A-Z0-9._-]*[A-Z0-9]|[A-Z0-9])(?P<extras> *\\[[^]]*])?(?P<constraints>[^;#]*)(?P<markers>[^#]*)(?P<comment>.*)$', re.IGNORECASE)
    matches = [(lineno, line, pattern.search(line)) for (lineno, line) in enumerate(lines, start=1)]
    requirements = []
    for (lineno, line, match) in matches:
        if not line.strip():
            continue
        if line.strip().startswith('#'):
            continue
        if line.startswith('git+https://'):
            continue
        requirements.append((lineno, line, match))
    return requirements
if __name__ == '__main__':
    main()