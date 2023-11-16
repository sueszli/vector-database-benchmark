import argparse
import html
import os
import re
import numpy as np
from typing import List
MIN_CHARS = 1
MAX_CHARS = 3000
dollar = re.compile('((?<!\\$)\\${1,2}(?!\\$))(.{%i,%i}?)(?<!\\\\)(?<!\\$)\\1(?!\\$)' % (1, MAX_CHARS))
inline = re.compile('(\\\\\\((.*?)(?<!\\\\)\\\\\\))|(\\\\\\[(.{%i,%i}?)(?<!\\\\)\\\\\\])' % (1, MAX_CHARS))
equation = re.compile('\\\\begin\\{(equation|math|displaymath)\\*?\\}(.{%i,%i}?)\\\\end\\{\\1\\*?\\}' % (1, MAX_CHARS), re.S)
align = re.compile('(\\\\begin\\{(align|alignedat|alignat|flalign|eqnarray|aligned|split|gather)\\*?\\}(.{%i,%i}?)\\\\end\\{\\2\\*?\\})' % (1, MAX_CHARS), re.S)
displaymath = re.compile('(?:\\\\displaystyle)(.{%i,%i}?)((?<!\\\\)\\}?(?:\\"|<))' % (1, MAX_CHARS), re.S)
outer_whitespace = re.compile('^\\\\,|\\\\,$|^~|~$|^\\\\ |\\\\ $|^\\\\thinspace|\\\\thinspace$|^\\\\!|\\\\!$|^\\\\:|\\\\:$|^\\\\;|\\\\;$|^\\\\enspace|\\\\enspace$|^\\\\quad|\\\\quad$|^\\\\qquad|\\\\qquad$|^\\\\hspace{[a-zA-Z0-9]+}|\\\\hspace{[a-zA-Z0-9]+}$|^\\\\hfill|\\\\hfill$')
label_names = [re.compile('\\\\%s\\s?\\{(.*?)\\}' % s) for s in ['ref', 'cite', 'label', 'eqref']]

def check_brackets(s):
    if False:
        print('Hello World!')
    a = []
    surrounding = False
    for (i, c) in enumerate(s):
        if c == '{':
            if i > 0 and s[i - 1] == '\\':
                continue
            else:
                a.append(1)
            if i == 0:
                surrounding = True
        elif c == '}':
            if i > 0 and s[i - 1] == '\\':
                continue
            else:
                a.append(-1)
    b = np.cumsum(a)
    if len(b) > 1 and b[-1] != 0:
        raise ValueError(s)
    surrounding = s[-1] == '}' and surrounding
    if not surrounding:
        return s
    elif (b == 0).sum() == 1:
        return s[1:-1]
    else:
        return s

def remove_labels(string):
    if False:
        return 10
    for s in label_names:
        string = re.sub(s, '', string)
    return string

def clean_matches(matches, min_chars=MIN_CHARS):
    if False:
        return 10
    faulty = []
    for i in range(len(matches)):
        if 'tikz' in matches[i]:
            faulty.append(i)
            continue
        matches[i] = remove_labels(matches[i])
        matches[i] = matches[i].replace('\n', '').replace('\\notag', '').replace('\\nonumber', '')
        matches[i] = re.sub(outer_whitespace, '', matches[i])
        if len(matches[i]) < min_chars:
            faulty.append(i)
            continue
        if matches[i][-1] == '\\' or 'newcommand' in matches[i][-1]:
            faulty.append(i)
    matches = [m.strip() for (i, m) in enumerate(matches) if i not in faulty]
    return list(set(matches))

def find_math(s: str, wiki=False) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Find all occurences of math in a Latex-like document. \n\n    Args:\n        s (str): String to search\n        wiki (bool, optional): Search for `\\displaystyle` as it can be found in the wikipedia page source code. Defaults to False.\n\n    Returns:\n        List[str]: List of all found mathematical expressions\n    '
    matches = []
    x = re.findall(inline, s)
    matches.extend([g[1] if g[1] != '' else g[-1] for g in x])
    if not wiki:
        patterns = [dollar, equation, align]
        groups = [1, 1, 0]
    else:
        patterns = [displaymath]
        groups = [0]
    for (i, pattern) in zip(groups, patterns):
        x = re.findall(pattern, s)
        matches.extend([g[i] for g in x])
    return clean_matches(matches)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='file', type=str, help='file to find equations in')
    parser.add_argument('--out', '-o', type=str, default=None, help='file to save equations to. If none provided, print all equations.')
    parser.add_argument('--wiki', action='store_true', help='only look for math starting with \\displaystyle')
    parser.add_argument('--unescape', action='store_true', help='call `html.unescape` on input')
    args = parser.parse_args()
    if not os.path.exists(args.file):
        raise ValueError('File can not be found. %s' % args.file)
    from pix2tex.dataset.demacro import pydemacro
    s = pydemacro(open(args.file, 'r', encoding='utf-8').read())
    if args.unescape:
        s = html.unescape(s)
    math = '\n'.join(sorted(find_math(s, args.wiki)))
    if args.out is None:
        print(math)
    else:
        with open(args.out, 'w') as f:
            f.write(math)