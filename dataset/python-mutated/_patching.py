"""
Write patches which add @example() decorators for discovered test cases.

Requires `hypothesis[codemods,ghostwriter]` installed, i.e. black and libcst.

This module is used by Hypothesis' builtin pytest plugin for failing examples
discovered during testing, and by HypoFuzz for _covering_ examples discovered
during fuzzing.
"""
import difflib
import hashlib
import inspect
import re
import sys
from ast import literal_eval
from contextlib import suppress
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import libcst as cst
from libcst import matchers as m
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
from hypothesis.configuration import storage_directory
from hypothesis.version import __version__
try:
    import black
except ImportError:
    black = None
HEADER = f'From HEAD Mon Sep 17 00:00:00 2001\nFrom: Hypothesis {__version__} <no-reply@hypothesis.works>\nDate: {{when:%a, %d %b %Y %H:%M:%S}}\nSubject: [PATCH] {{msg}}\n\n---\n'
FAIL_MSG = 'discovered failure'
_space_only_re = re.compile('^ +$', re.MULTILINE)
_leading_space_re = re.compile('(^[ ]*)(?:[^ \n])', re.MULTILINE)

def dedent(text):
    if False:
        for i in range(10):
            print('nop')
    text = _space_only_re.sub('', text)
    prefix = min(_leading_space_re.findall(text), key=len)
    return (re.sub('(?m)^' + prefix, '', text), prefix)

def indent(text: str, prefix: str) -> str:
    if False:
        i = 10
        return i + 15
    return ''.join((prefix + line for line in text.splitlines(keepends=True)))

class AddExamplesCodemod(VisitorBasedCodemodCommand):
    DESCRIPTION = 'Add explicit examples to failing tests.'

    def __init__(self, context, fn_examples, strip_via=(), dec='example', width=88):
        if False:
            print('Hello World!')
        'Add @example() decorator(s) for failing test(s).\n\n        `code` is the source code of the module where the test functions are defined.\n        `fn_examples` is a dict of function name to list-of-failing-examples.\n        '
        assert fn_examples, 'This codemod does nothing without fn_examples.'
        super().__init__(context)
        self.decorator_func = cst.parse_expression(dec)
        self.line_length = width
        value_in_strip_via = m.MatchIfTrue(lambda x: literal_eval(x.value) in strip_via)
        self.strip_matching = m.Call(m.Attribute(m.Call(), m.Name('via')), [m.Arg(m.SimpleString() & value_in_strip_via)])
        self.fn_examples = {k: tuple((self.__call_node_to_example_dec(ex, via) for (ex, via) in nodes)) for (k, nodes) in fn_examples.items()}

    def __call_node_to_example_dec(self, node, via):
        if False:
            for i in range(10):
                print('nop')
        node = node.with_changes(func=self.decorator_func, args=[a.with_changes(comma=a.comma if m.findall(a.comma, m.Comment()) else cst.MaybeSentinel.DEFAULT) for a in node.args] if black else node.args)
        via = cst.Call(func=cst.Attribute(node, cst.Name('via')), args=[cst.Arg(cst.SimpleString(repr(via)))])
        if black:
            pretty = black.format_str(cst.Module([]).code_for_node(via), mode=black.FileMode(line_length=self.line_length))
            via = cst.parse_expression(pretty.strip())
        return cst.Decorator(via)

    def leave_FunctionDef(self, _, updated_node):
        if False:
            return 10
        return updated_node.with_changes(decorators=tuple((d for d in updated_node.decorators if not m.findall(d, self.strip_matching))) + self.fn_examples.get(updated_node.name.value, ()))

def get_patch_for(func, failing_examples, *, strip_via=()):
    if False:
        while True:
            i = 10
    try:
        module = sys.modules[func.__module__]
        fname = Path(module.__file__).relative_to(Path.cwd())
        before = inspect.getsource(func)
    except Exception:
        return None
    call_nodes = []
    for (ex, via) in set(failing_examples):
        with suppress(Exception):
            node = cst.parse_expression(ex)
            assert isinstance(node, cst.Call), node
            data = m.Arg(m.Call(m.Name('data'), args=[m.Arg(m.Ellipsis())]))
            if m.matches(node, m.Call(args=[m.ZeroOrMore(), data, m.ZeroOrMore()])):
                return None
            call_nodes.append((node, via))
    if not call_nodes:
        return None
    if module.__dict__.get('hypothesis') is sys.modules['hypothesis'] and 'given' not in module.__dict__:
        decorator_func = 'hypothesis.example'
    else:
        decorator_func = 'example'
    (dedented, prefix) = dedent(before)
    try:
        node = cst.parse_module(dedented)
    except Exception:
        return None
    after = AddExamplesCodemod(CodemodContext(), fn_examples={func.__name__: call_nodes}, strip_via=strip_via, dec=decorator_func, width=88 - len(prefix)).transform_module(node)
    return (str(fname), before, indent(after.code, prefix=prefix))

def make_patch(triples, *, msg='Hypothesis: add explicit examples', when=None):
    if False:
        i = 10
        return i + 15
    'Create a patch for (fname, before, after) triples.'
    assert triples, 'attempted to create empty patch'
    when = when or datetime.now(tz=timezone.utc)
    by_fname = {}
    for (fname, before, after) in triples:
        by_fname.setdefault(Path(fname), []).append((before, after))
    diffs = [HEADER.format(msg=msg, when=when)]
    for (fname, changes) in sorted(by_fname.items()):
        source_before = source_after = fname.read_text(encoding='utf-8')
        for (before, after) in changes:
            source_after = source_after.replace(before.rstrip(), after.rstrip(), 1)
        ud = difflib.unified_diff(source_before.splitlines(keepends=True), source_after.splitlines(keepends=True), fromfile=str(fname), tofile=str(fname))
        diffs.append(''.join(ud))
    return ''.join(diffs)

def save_patch(patch: str, *, slug: str='') -> Path:
    if False:
        i = 10
        return i + 15
    assert re.fullmatch('|[a-z]+-', slug), f'malformed slug={slug!r}'
    now = date.today().isoformat()
    cleaned = re.sub('^Date: .+?$', '', patch, count=1, flags=re.MULTILINE)
    hash8 = hashlib.sha1(cleaned.encode()).hexdigest()[:8]
    fname = Path(storage_directory('patches', f'{now}--{slug}{hash8}.patch'))
    fname.parent.mkdir(parents=True, exist_ok=True)
    fname.write_text(patch, encoding='utf-8')
    return fname.relative_to(Path.cwd())

def gc_patches(slug: str='') -> None:
    if False:
        print('Hello World!')
    cutoff = date.today() - timedelta(days=7)
    for fname in Path(storage_directory('patches')).glob(f'????-??-??--{slug}????????.patch'):
        if date.fromisoformat(fname.stem.split('--')[0]) < cutoff:
            fname.unlink()