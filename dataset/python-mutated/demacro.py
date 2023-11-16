import argparse
import re
import logging
from collections import Counter
import time
from pix2tex.dataset.extract_latex import remove_labels

class DemacroError(Exception):
    pass

def main():
    if False:
        for i in range(10):
            print('nop')
    args = parse_command_line()
    data = read(args.input)
    data = pydemacro(data)
    if args.output is not None:
        write(args.output, data)
    else:
        print(data)

def parse_command_line():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Replace \\def with \\newcommand where possible.')
    parser.add_argument('input', help='TeX input file with \\def')
    parser.add_argument('--output', '-o', default=None, help='TeX output file with \\newcommand')
    return parser.parse_args()

def read(path):
    if False:
        print('Hello World!')
    with open(path, mode='r') as handle:
        return handle.read()

def bracket_replace(string: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    replaces all layered brackets with special symbols\n    '
    layer = 0
    out = list(string)
    for (i, c) in enumerate(out):
        if c == '{':
            if layer > 0:
                out[i] = 'Ḋ'
            layer += 1
        elif c == '}':
            layer -= 1
            if layer > 0:
                out[i] = 'Ḍ'
    return ''.join(out)

def undo_bracket_replace(string):
    if False:
        for i in range(10):
            print('nop')
    return string.replace('Ḋ', '{').replace('Ḍ', '}')

def sweep(t, cmds):
    if False:
        return 10
    num_matches = 0
    for c in cmds:
        nargs = int(c[1][1]) if c[1] != '' else 0
        optional = c[2] != ''
        if nargs == 0:
            num_matches += len(re.findall('\\\\%s([\\W_^\\dĊ])' % c[0], t))
            if num_matches > 0:
                t = re.sub('\\\\%s([\\W_^\\dĊ])' % c[0], '%s\\1' % c[-1].replace('\\', '\\\\'), t)
        else:
            matches = re.findall('(\\\\%s(?:\\[(.+?)\\])?' % c[0] + '{(.+?)}' * (nargs - (1 if optional else 0)) + ')', t)
            num_matches += len(matches)
            for (i, m) in enumerate(matches):
                r = c[-1]
                if m[1] == '':
                    matches[i] = (m[0], c[2][1:-1], *m[2:])
                for j in range(1, nargs + 1):
                    r = r.replace('#%i' % j, matches[i][j + int(not optional)])
                t = t.replace(matches[i][0], r)
    return (t, num_matches)

def unfold(t):
    if False:
        while True:
            i = 10
    t = t.replace('\n', 'Ċ')
    t = bracket_replace(t)
    commands_pattern = '\\\\(?:re)?newcommand\\*?{\\\\(.+?)}[\\sĊ]*(\\[\\d\\])?[\\sĊ]*(\\[.+?\\])?[\\sĊ]*{(.*?)}'
    cmds = re.findall(commands_pattern, t)
    t = re.sub('(?<!\\\\)' + commands_pattern, 'Ċ', t)
    cmds = sorted(cmds, key=lambda x: len(x[0]))
    cmd_names = Counter([c[0] for c in cmds])
    for i in reversed(range(len(cmds))):
        if cmd_names[cmds[i][0]] > 1:
            del cmds[i]
        elif '\\newcommand' in cmds[i][-1]:
            logging.debug("Command recognition pattern didn't work properly. %s" % undo_bracket_replace(cmds[i][-1]))
            del cmds[i]
    start = time.time()
    try:
        for i in range(10):
            if i > 0:
                t = bracket_replace(t)
            (t, N) = sweep(t, cmds)
            if time.time() - start > 5:
                raise TimeoutError
            t = undo_bracket_replace(t)
            if N == 0 or i == 9:
                break
            elif N > 4000:
                raise ValueError('Too many matches. Processing would take too long.')
    except ValueError:
        pass
    except TimeoutError:
        pass
    except re.error as e:
        raise DemacroError(e)
    t = remove_labels(t.replace('Ċ', '\n'))
    return t

def pydemacro(t: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Replaces all occurences of newly defined Latex commands in a document.\n    Can replace `\\newcommand`, `\\def` and `\\let` definitions in the code.\n\n    Args:\n        t (str): Latex document\n\n    Returns:\n        str: Document without custom commands\n    '
    return unfold(convert(re.sub('\n+', '\n', re.sub('(?<!\\\\)%.*\\n', '\n', t))))

def replace(match):
    if False:
        print('Hello World!')
    prefix = match.group(1)
    if prefix is not None and ('expandafter' in prefix or 'global' in prefix or 'outer' in prefix or ('protected' in prefix)):
        return match.group(0)
    result = '\\newcommand'
    if prefix is None or 'long' not in prefix:
        result += '*'
    result += '{' + match.group(2) + '}'
    if match.lastindex == 3:
        result += '[' + match.group(3) + ']'
    result += '{'
    return result

def convert(data):
    if False:
        for i in range(10):
            print('nop')
    data = re.sub('((?:\\\\(?:expandafter|global|long|outer|protected)(?:\\s+|\\r?\\n\\s*)?)*)?\\\\def\\s*(\\\\[a-zA-Z]+)\\s*(?:#+([0-9]))*\\{', replace, data)
    return re.sub('\\\\let[\\sĊ]*(\\\\[a-zA-Z]+)\\s*=?[\\sĊ]*(\\\\?\\w+)*', '\\\\newcommand*{\\1}{\\2}\\n', data)

def write(path, data):
    if False:
        return 10
    with open(path, mode='w') as handle:
        handle.write(data)
    print('=> File written: {0}'.format(path))
if __name__ == '__main__':
    main()