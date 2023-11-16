"""A super simple linter to check C syntax."""
from __future__ import print_function
import argparse
import sys
warned = False

def warn(path, line, lineno, msg):
    if False:
        print('Hello World!')
    global warned
    warned = True
    print('%s:%s: %s' % (path, lineno, msg), file=sys.stderr)

def check_line(path, line, idx, lines):
    if False:
        for i in range(10):
            print('nop')
    s = line
    lineno = idx + 1
    eof = lineno == len(lines)
    if s.endswith(' \n'):
        warn(path, line, lineno, 'extra space at EOL')
    elif '\t' in line:
        warn(path, line, lineno, 'line has a tab')
    elif s.endswith('\r\n'):
        warn(path, line, lineno, 'Windows line ending')
    elif s == '}\n':
        if not eof:
            nextline = lines[idx + 1]
            if nextline != '\n' and nextline.strip()[0] != '#' and (nextline.strip()[:2] != '*/'):
                warn(path, line, lineno, 'expected 1 blank line')
    sls = s.lstrip()
    if sls.startswith('//') and sls[2] != ' ' and (line.strip() != '//'):
        warn(path, line, lineno, 'no space after // comment')
    keywords = ('if', 'else', 'while', 'do', 'enum', 'for')
    for kw in keywords:
        if sls.startswith(kw + '('):
            warn(path, line, lineno, "missing space between %r and '('" % kw)
    if eof and (not line.endswith('\n')):
        warn(path, line, lineno, 'no blank line at EOF')
    ss = s.strip()
    if ss.startswith(('printf(', 'printf (')):
        if not ss.endswith(('// NOQA', '//  NOQA')):
            warn(path, line, lineno, 'printf() statement')

def process(path):
    if False:
        i = 10
        return i + 15
    with open(path) as f:
        lines = f.readlines()
    for (idx, line) in enumerate(lines):
        check_line(path, line, idx, lines)

def main():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', help='path(s) to a file(s)')
    args = parser.parse_args()
    for path in args.paths:
        process(path)
    if warned:
        sys.exit(1)
if __name__ == '__main__':
    main()