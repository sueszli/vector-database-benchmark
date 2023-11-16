from __future__ import annotations
import ast
from typing import Iterable
from tokenize_rt import Offset
from tokenize_rt import parse_string_literal
from tokenize_rt import Token
from tokenize_rt import tokens_to_src
from pyupgrade._ast_helpers import ast_to_offset
from pyupgrade._ast_helpers import contains_await
from pyupgrade._ast_helpers import has_starargs
from pyupgrade._data import register
from pyupgrade._data import State
from pyupgrade._data import TokenFunc
from pyupgrade._string_helpers import parse_format
from pyupgrade._string_helpers import unparse_parsed_string
from pyupgrade._token_helpers import parse_call_args

def _skip_unimportant_ws(tokens: list[Token], i: int) -> int:
    if False:
        while True:
            i = 10
    while tokens[i].name == 'UNIMPORTANT_WS':
        i += 1
    return i

def _to_fstring(src: str, tokens: list[Token], args: list[tuple[int, int]]) -> str:
    if False:
        i = 10
        return i + 15
    params = {}
    i = 0
    for (start, end) in args:
        start = _skip_unimportant_ws(tokens, start)
        if tokens[start].name == 'NAME':
            after = _skip_unimportant_ws(tokens, start + 1)
            if tokens[after].src == '=':
                params[tokens[start].src] = tokens_to_src(tokens[after + 1:end]).strip()
                continue
        params[str(i)] = tokens_to_src(tokens[start:end]).strip()
        i += 1
    parts = []
    i = 0
    (prefix, rest) = parse_string_literal(src)
    new_src = 'f' + prefix.translate({ord('u'): None, ord('U'): None}) + rest
    for (s, name, spec, conv) in parse_format(new_src):
        if name is not None:
            (k, dot, rest) = name.partition('.')
            name = ''.join((params[k or str(i)], dot, rest))
            if not k:
                i += 1
        parts.append((s, name, spec, conv))
    return unparse_parsed_string(parts)

def _fix_fstring(i: int, tokens: list[Token]) -> None:
    if False:
        return 10
    token = tokens[i]
    paren = i + 3
    if tokens_to_src(tokens[i + 1:paren + 1]) != '.format(':
        return
    (args, end) = parse_call_args(tokens, paren)
    if tokens[end - 1].line != token.line:
        return
    args_src = tokens_to_src(tokens[paren:end])
    if '\\' in args_src or '"' in args_src or "'" in args_src:
        return
    tokens[i] = token._replace(src=_to_fstring(token.src, tokens, args))
    del tokens[i + 1:end]

def _format_params(call: ast.Call) -> set[str]:
    if False:
        for i in range(10):
            print('nop')
    params = {str(i) for (i, arg) in enumerate(call.args)}
    for kwd in call.keywords:
        assert kwd.arg is not None
        params.add(kwd.arg)
    return params

@register(ast.Call)
def visit_Call(state: State, node: ast.Call, parent: ast.AST) -> Iterable[tuple[Offset, TokenFunc]]:
    if False:
        return 10
    if state.settings.min_version < (3, 6):
        return
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str) and (node.func.attr == 'format') and (not has_starargs(node)):
        try:
            parsed = parse_format(node.func.value.value)
        except ValueError:
            return
        params = _format_params(node)
        seen = set()
        i = 0
        for (_, name, spec, _) in parsed:
            if spec is not None and '{' in spec:
                break
            if name is not None:
                (candidate, _, _) = name.partition('.')
                if candidate and candidate in seen:
                    break
                elif '[' in name:
                    break
                seen.add(candidate)
                key = candidate or str(i)
                if key not in params:
                    break
                if not candidate:
                    i += 1
        else:
            if state.settings.min_version >= (3, 7) or not contains_await(node):
                yield (ast_to_offset(node), _fix_fstring)