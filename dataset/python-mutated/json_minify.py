"""A port of the `JSON-minify` utility to the Python language.

Based on JSON.minify.js: https://github.com/getify/JSON.minify

Contributors:
  - Gerald Storer
    - Contributed original version
  - Felipe Machado
    - Performance optimization
  - Pradyun S. Gedam
    - Conditions and variable names changed
    - Reformatted tests and moved to separate file
    - Made into a PyPI Package
"""
import re

def minify(string, strip_space=False):
    if False:
        i = 10
        return i + 15
    tokenizer = re.compile('"|(/\\*)|(\\*/)|(//)|\n|\r')
    end_slashes_re = re.compile('(\\\\)*$')
    in_string = False
    in_multi = False
    in_single = False
    new_str = []
    index = 0
    for match in re.finditer(tokenizer, string):
        if not (in_multi or in_single):
            tmp = string[index:match.start()]
            if not in_string and strip_space:
                tmp = re.sub('[ \t\n\r]+', '', tmp)
            new_str.append(tmp)
        elif not strip_space:
            new_str.append(' ' * (match.start() - index))
        index = match.end()
        val = match.group()
        if val == '"' and (not (in_multi or in_single)):
            escaped = end_slashes_re.search(string, 0, match.start())
            if not in_string or (escaped is None or len(escaped.group()) % 2 == 0):
                in_string = not in_string
            index -= 1
        elif not (in_string or in_multi or in_single):
            if val == '/*':
                in_multi = True
            elif val == '//':
                in_single = True
        elif val == '*/' and in_multi and (not (in_string or in_single)):
            in_multi = False
            if not strip_space:
                new_str.append(' ' * len(val))
        elif val in '\r\n' and (not (in_multi or in_string)) and in_single:
            in_single = False
        elif not ((in_multi or in_single) or (val in ' \r\n\t' and strip_space)):
            new_str.append(val)
        if not strip_space:
            if val in '\r\n':
                new_str.append(val)
            elif in_multi or in_single:
                new_str.append(' ' * len(val))
    new_str.append(string[index:])
    return ''.join(new_str)