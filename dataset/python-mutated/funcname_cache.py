import tokenize
from typing import Dict, List, Optional
cache: Dict[str, Dict[int, str]] = {}

def clearcache() -> None:
    if False:
        while True:
            i = 10
    cache.clear()

def _add_file(filename: str) -> None:
    if False:
        i = 10
        return i + 15
    try:
        with open(filename) as f:
            tokens = list(tokenize.generate_tokens(f.readline))
    except OSError:
        cache[filename] = {}
        return
    result: Dict[int, str] = {}
    cur_name = ''
    cur_indent = 0
    significant_indents: List[int] = []
    for (i, token) in enumerate(tokens):
        if token.type == tokenize.INDENT:
            cur_indent += 1
        elif token.type == tokenize.DEDENT:
            cur_indent -= 1
            if significant_indents and cur_indent == significant_indents[-1]:
                significant_indents.pop()
                cur_name = cur_name.rpartition('.')[0]
        elif token.type == tokenize.NAME and i + 1 < len(tokens) and (tokens[i + 1].type == tokenize.NAME) and (token.string == 'class' or token.string == 'def'):
            significant_indents.append(cur_indent)
            if cur_name:
                cur_name += '.'
            cur_name += tokens[i + 1].string
        result[token.start[0]] = cur_name
    cache[filename] = result

def get_funcname(filename: str, lineno: int) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    if filename not in cache:
        _add_file(filename)
    return cache[filename].get(lineno, None)