from __future__ import annotations
import re
import argparse
import textwrap
import contextlib
from typing import Match, Optional, Sequence, Generator, NamedTuple, cast
import black
from black.mode import TargetVersion
from black.const import DEFAULT_LINE_LENGTH
MD_RE = re.compile('(?P<before>^(?P<indent> *)```\\s*python\\n)(?P<code>.*?)(?P<after>^(?P=indent)```\\s*$)', re.DOTALL | re.MULTILINE)
MD_PYCON_RE = re.compile('(?P<before>^(?P<indent> *)```\\s*pycon\\n)(?P<code>.*?)(?P<after>^(?P=indent)```.*$)', re.DOTALL | re.MULTILINE)
RST_PY_LANGS = frozenset(('python', 'py', 'sage', 'python3', 'py3', 'numpy'))
BLOCK_TYPES = '(code|code-block|sourcecode|ipython)'
DOCTEST_TYPES = '(testsetup|testcleanup|testcode)'
RST_RE = re.compile(f'(?P<before>^(?P<indent> *)\\.\\. (jupyter-execute::|{BLOCK_TYPES}:: (?P<lang>\\w+)|{DOCTEST_TYPES}::.*)\\n((?P=indent) +:.*\\n)*\\n*)(?P<code>(^((?P=indent) +.*)?\\n)+)', re.MULTILINE)
RST_PYCON_RE = re.compile('(?P<before>(?P<indent> *)\\.\\. ((code|code-block):: pycon|doctest::.*)\\n((?P=indent) +:.*\\n)*\\n*)(?P<code>(^((?P=indent) +.*)?(\\n|$))+)', re.MULTILINE)
PYCON_PREFIX = '>>> '
PYCON_CONTINUATION_PREFIX = '...'
PYCON_CONTINUATION_RE = re.compile(f'^{re.escape(PYCON_CONTINUATION_PREFIX)}( |$)')
LATEX_RE = re.compile('(?P<before>^(?P<indent> *)\\\\begin{minted}{python}\\n)(?P<code>.*?)(?P<after>^(?P=indent)\\\\end{minted}\\s*$)', re.DOTALL | re.MULTILINE)
LATEX_PYCON_RE = re.compile('(?P<before>^(?P<indent> *)\\\\begin{minted}{pycon}\\n)(?P<code>.*?)(?P<after>^(?P=indent)\\\\end{minted}\\s*$)', re.DOTALL | re.MULTILINE)
PYTHONTEX_LANG = '(?P<lang>pyblock|pycode|pyconsole|pyverbatim)'
PYTHONTEX_RE = re.compile(f'(?P<before>^(?P<indent> *)\\\\begin{{{PYTHONTEX_LANG}}}\\n)(?P<code>.*?)(?P<after>^(?P=indent)\\\\end{{(?P=lang)}}\\s*$)', re.DOTALL | re.MULTILINE)
INDENT_RE = re.compile('^ +(?=[^ ])', re.MULTILINE)
TRAILING_NL_RE = re.compile('\\n+\\Z', re.MULTILINE)

class CodeBlockError(NamedTuple):
    offset: int
    exc: Exception

def format_str(src: str, black_mode: black.FileMode) -> tuple[str, Sequence[CodeBlockError]]:
    if False:
        for i in range(10):
            print('nop')
    errors: list[CodeBlockError] = []

    @contextlib.contextmanager
    def _collect_error(match: Match[str]) -> Generator[None, None, None]:
        if False:
            i = 10
            return i + 15
        try:
            yield
        except Exception as e:
            errors.append(CodeBlockError(match.start(), e))

    def _md_match(match: Match[str]) -> str:
        if False:
            while True:
                i = 10
        code = textwrap.dedent(match['code'])
        with _collect_error(match):
            code = black.format_str(code, mode=black_mode)
        code = textwrap.indent(code, match['indent'])
        return f"{match['before']}{code}{match['after']}"

    def _rst_match(match: Match[str]) -> str:
        if False:
            i = 10
            return i + 15
        lang = match['lang']
        if lang is not None and lang not in RST_PY_LANGS:
            return match[0]
        min_indent = min(INDENT_RE.findall(match['code']))
        trailing_ws_match = TRAILING_NL_RE.search(match['code'])
        assert trailing_ws_match
        trailing_ws = trailing_ws_match.group()
        code = textwrap.dedent(match['code'])
        with _collect_error(match):
            code = black.format_str(code, mode=black_mode)
        code = textwrap.indent(code, min_indent)
        return f"{match['before']}{code.rstrip()}{trailing_ws}"

    def _pycon_match(match: Match[str]) -> str:
        if False:
            for i in range(10):
                print('nop')
        code = ''
        fragment = cast(Optional[str], None)

        def finish_fragment() -> None:
            if False:
                i = 10
                return i + 15
            nonlocal code
            nonlocal fragment
            if fragment is not None:
                with _collect_error(match):
                    fragment = black.format_str(fragment, mode=black_mode)
                fragment_lines = fragment.splitlines()
                code += f'{PYCON_PREFIX}{fragment_lines[0]}\n'
                for line in fragment_lines[1:]:
                    if line:
                        code += f'{PYCON_CONTINUATION_PREFIX} {line}\n'
                if fragment_lines[-1].startswith(' '):
                    code += f'{PYCON_CONTINUATION_PREFIX}\n'
                fragment = None
        indentation = None
        for line in match['code'].splitlines():
            (orig_line, line) = (line, line.lstrip())
            if indentation is None and line:
                indentation = len(orig_line) - len(line)
            continuation_match = PYCON_CONTINUATION_RE.match(line)
            if continuation_match and fragment is not None:
                fragment += line[continuation_match.end():] + '\n'
            else:
                finish_fragment()
                if line.startswith(PYCON_PREFIX):
                    fragment = line[len(PYCON_PREFIX):] + '\n'
                else:
                    code += orig_line[indentation:] + '\n'
        finish_fragment()
        return code

    def _md_pycon_match(match: Match[str]) -> str:
        if False:
            for i in range(10):
                print('nop')
        code = _pycon_match(match)
        code = textwrap.indent(code, match['indent'])
        return f"{match['before']}{code}{match['after']}"

    def _rst_pycon_match(match: Match[str]) -> str:
        if False:
            print('Hello World!')
        code = _pycon_match(match)
        min_indent = min(INDENT_RE.findall(match['code']))
        code = textwrap.indent(code, min_indent)
        return f"{match['before']}{code}"

    def _latex_match(match: Match[str]) -> str:
        if False:
            while True:
                i = 10
        code = textwrap.dedent(match['code'])
        with _collect_error(match):
            code = black.format_str(code, mode=black_mode)
        code = textwrap.indent(code, match['indent'])
        return f"{match['before']}{code}{match['after']}"

    def _latex_pycon_match(match: Match[str]) -> str:
        if False:
            while True:
                i = 10
        code = _pycon_match(match)
        code = textwrap.indent(code, match['indent'])
        return f"{match['before']}{code}{match['after']}"
    src = MD_RE.sub(_md_match, src)
    src = MD_PYCON_RE.sub(_md_pycon_match, src)
    src = RST_RE.sub(_rst_match, src)
    src = RST_PYCON_RE.sub(_rst_pycon_match, src)
    src = LATEX_RE.sub(_latex_match, src)
    src = LATEX_PYCON_RE.sub(_latex_pycon_match, src)
    src = PYTHONTEX_RE.sub(_latex_match, src)
    return (src, errors)

def format_file(filename: str, black_mode: black.FileMode, skip_errors: bool) -> int:
    if False:
        print('Hello World!')
    with open(filename, encoding='UTF-8') as f:
        contents = f.read()
    (new_contents, errors) = format_str(contents, black_mode)
    for error in errors:
        lineno = contents[:error.offset].count('\n') + 1
        print(f'{filename}:{lineno}: code block parse error {error.exc}')
    if errors and (not skip_errors):
        return 1
    if contents != new_contents:
        print(f'{filename}: Rewriting...')
        with open(filename, 'w', encoding='UTF-8') as f:
            f.write(new_contents)
        return 0
    else:
        return 0

def main(argv: Sequence[str] | None=None) -> int:
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--line-length', type=int, default=DEFAULT_LINE_LENGTH)
    parser.add_argument('-t', '--target-version', action='append', type=lambda v: TargetVersion[v.upper()], default=[], help=f'choices: {[v.name.lower() for v in TargetVersion]}', dest='target_versions')
    parser.add_argument('-S', '--skip-string-normalization', action='store_true')
    parser.add_argument('-E', '--skip-errors', action='store_true')
    parser.add_argument('filenames', nargs='*')
    args = parser.parse_args(argv)
    black_mode = black.FileMode(target_versions=set(args.target_versions), line_length=args.line_length, string_normalization=not args.skip_string_normalization)
    retv = 0
    for filename in args.filenames:
        retv |= format_file(filename, black_mode, skip_errors=args.skip_errors)
    return retv
if __name__ == '__main__':
    raise SystemExit(main())