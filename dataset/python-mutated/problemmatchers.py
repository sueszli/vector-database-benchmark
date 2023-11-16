"""Register problem matchers for GitHub Actions.

Relevant docs:
https://github.com/actions/toolkit/blob/master/docs/problem-matchers.md
https://github.com/actions/toolkit/blob/master/docs/commands.md#problem-matchers
"""
import sys
import pathlib
import json
MATCHERS = {'shellcheck': [{'pattern': [{'regexp': '^(.+):(\\d+):(\\d+):\\s(note|warning|error):\\s(.*)\\s\\[(SC\\d+)\\]$', 'file': 1, 'line': 2, 'column': 3, 'severity': 4, 'message': 5, 'code': 6}]}], 'yamllint': [{'pattern': [{'regexp': '^\\033\\[4m([^\\033]+)\\033\\[0m$', 'file': 1}, {'regexp': '^  \\033\\[2m(\\d+):(\\d+)\\033\\[0m   \\033\\[3[13]m([^\\033]+)\\033\\[0m +([^\\033]*)\\033\\[2m\\(([^)]+)\\)\\033\\[0m$', 'line': 1, 'column': 2, 'severity': 3, 'message': 4, 'code': 5, 'loop': True}]}], 'actionlint': [{'pattern': [{'regexp': '^(?:\\x1b\\[\\d+m)?(.+?)(?:\\x1b\\[\\d+m)*:(?:\\x1b\\[\\d+m)*(\\d+)(?:\\x1b\\[\\d+m)*:(?:\\x1b\\[\\d+m)*(\\d+)(?:\\x1b\\[\\d+m)*: (?:\\x1b\\[\\d+m)*(.+?)(?:\\x1b\\[\\d+m)* \\[(.+?)\\]$', 'file': 1, 'line': 2, 'column': 3, 'message': 4, 'code': 5}]}], 'vulture': [{'severity': 'warning', 'pattern': [{'regexp': '^([^:]+):(\\d+): ([^(]+ \\(\\d+% confidence\\))$', 'file': 1, 'line': 2, 'message': 3}]}], 'flake8': [{'severity': 'warning', 'pattern': [{'regexp': '^(\\033\\[0m)?([^:]+):(\\d+):(\\d+): ([A-Z]\\d{3}) (.*)$', 'file': 2, 'line': 3, 'column': 4, 'code': 5, 'message': 6}]}], 'mypy': [{'pattern': [{'regexp': '^(\\033\\[0m)?([^:]+):(\\d+): ([^:]+): (.*)  \\[(.*)\\]$', 'file': 2, 'line': 3, 'severity': 4, 'message': 5, 'code': 6}]}], 'pylint': [{'severity': 'error', 'pattern': [{'regexp': '^([^:]+):(\\d+):(\\d+): (E\\d+): \\033\\[[\\d;]+m([^\\033]+).*$', 'file': 1, 'line': 2, 'column': 3, 'code': 4, 'message': 5}]}, {'severity': 'warning', 'pattern': [{'regexp': '^([^:]+):(\\d+):(\\d+): ([A-DF-Z]\\d+): \\033\\[[\\d;]+m([^\\033]+).*$', 'file': 1, 'line': 2, 'column': 3, 'code': 4, 'message': 5}]}], 'tests': [{'severity': 'error', 'pattern': [{'regexp': '^=+ short test summary info =+$'}, {'regexp': '^((ERROR|FAILED) .*)', 'message': 1, 'loop': True}]}, {'severity': 'error', 'pattern': [{'regexp': '^\\033\\[1m\\033\\[31mE       ([a-zA-Z0-9.]+: [^\\033]*)\\033\\[0m$', 'message': 1}]}], 'misc': [{'severity': 'error', 'pattern': [{'regexp': '^([^:]+):(\\d+): \\033\\[34m(Found .*)\\033\\[0m', 'file': 1, 'line': 2, 'message': 3}]}]}

def add_matcher(output_dir, owner, data):
    if False:
        while True:
            i = 10
    data['owner'] = owner
    out_data = {'problemMatcher': [data]}
    output_file = output_dir / '{}.json'.format(owner)
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(out_data, f)
    print('::add-matcher::{}'.format(output_file))

def main(testenv, tempdir):
    if False:
        print('Hello World!')
    testenv = sys.argv[1]
    if testenv.startswith('py3'):
        testenv = 'tests'
    if testenv not in MATCHERS:
        return
    output_dir = pathlib.Path(tempdir)
    for (idx, data) in enumerate(MATCHERS[testenv]):
        owner = '{}-{}'.format(testenv, idx)
        add_matcher(output_dir=output_dir, owner=owner, data=data)
if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))