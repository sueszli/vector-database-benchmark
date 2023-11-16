"""This file can approximately be considered the collection of hypothesis going
to really unreasonable lengths to produce pretty output."""
import ast
import hashlib
import inspect
import os
import re
import sys
import textwrap
import types
from functools import wraps
from io import StringIO
from keyword import iskeyword
from tokenize import COMMENT, detect_encoding, generate_tokens, untokenize
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable
from unittest.mock import _patch as PatchType
from hypothesis.internal.compat import PYPY, is_typed_named_tuple
from hypothesis.utils.conventions import not_set
from hypothesis.vendor.pretty import pretty
if TYPE_CHECKING:
    from hypothesis.strategies._internal.strategies import T
READTHEDOCS = os.environ.get('READTHEDOCS', None) == 'True'

def is_mock(obj):
    if False:
        while True:
            i = 10
    'Determine if the given argument is a mock type.'
    return hasattr(obj, 'hypothesis_internal_is_this_a_mock_check')

def _clean_source(src: str) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    "Return the source code as bytes, without decorators or comments.\n\n    Because this is part of our database key, we reduce the cache invalidation\n    rate by ignoring decorators, comments, trailing whitespace, and empty lines.\n    We can't just use the (dumped) AST directly because it changes between Python\n    versions (e.g. ast.Constant)\n    "
    try:
        funcdef = ast.parse(src).body[0]
        if sys.version_info[:2] == (3, 8) and PYPY:
            tag = 'async def ' if isinstance(funcdef, ast.AsyncFunctionDef) else 'def '
            if tag in src:
                src = tag + src.split(tag, maxsplit=1)[1]
        else:
            src = ''.join(src.splitlines(keepends=True)[funcdef.lineno - 1:])
    except Exception:
        pass
    try:
        src = untokenize((t for t in generate_tokens(StringIO(src).readline) if t.type != COMMENT))
    except Exception:
        pass
    return '\n'.join((x.rstrip() for x in src.splitlines() if x.rstrip())).encode()

def function_digest(function):
    if False:
        i = 10
        return i + 15
    'Returns a string that is stable across multiple invocations across\n    multiple processes and is prone to changing significantly in response to\n    minor changes to the function.\n\n    No guarantee of uniqueness though it usually will be. Digest collisions\n    lead to unfortunate but not fatal problems during database replay.\n    '
    hasher = hashlib.sha384()
    try:
        src = inspect.getsource(function)
    except (OSError, TypeError):
        try:
            hasher.update(function.__name__.encode())
        except AttributeError:
            pass
    else:
        hasher.update(_clean_source(src))
    try:
        hasher.update(repr(get_signature(function)).encode())
    except Exception:
        pass
    try:
        hasher.update(function._hypothesis_internal_add_digest)
    except AttributeError:
        pass
    return hasher.digest()

def check_signature(sig: inspect.Signature) -> None:
    if False:
        return 10
    for p in sig.parameters.values():
        if iskeyword(p.name) and p.kind is not p.POSITIONAL_ONLY:
            raise ValueError(f'Signature {sig!r} contains a parameter named {p.name!r}, but this is a SyntaxError because `{p.name}` is a keyword. You, or a library you use, must have manually created an invalid signature - this will be an error in Python 3.11+')

def get_signature(target: Any, *, follow_wrapped: bool=True, eval_str: bool=False) -> inspect.Signature:
    if False:
        return 10
    patches = getattr(target, 'patchings', None)
    if isinstance(patches, list) and all((isinstance(p, PatchType) for p in patches)):
        P = inspect.Parameter
        return inspect.Signature([P('args', P.VAR_POSITIONAL), P('keywargs', P.VAR_KEYWORD)])
    if isinstance(getattr(target, '__signature__', None), inspect.Signature):
        sig = target.__signature__
        check_signature(sig)
        if sig.parameters and (inspect.isclass(target) or inspect.ismethod(target)):
            selfy = next(iter(sig.parameters.values()))
            if selfy.name == 'self' and selfy.default is inspect.Parameter.empty and selfy.kind.name.startswith('POSITIONAL_'):
                return sig.replace(parameters=[v for (k, v) in sig.parameters.items() if k != 'self'])
        return sig
    if sys.version_info[:2] <= (3, 8) and inspect.isclass(target):
        from hypothesis.strategies._internal.types import is_generic_type
        if is_generic_type(target):
            sig = inspect.signature(target.__init__)
            check_signature(sig)
            return sig.replace(parameters=[v for (k, v) in sig.parameters.items() if k != 'self'])
    if sys.version_info[:2] >= (3, 10):
        sig = inspect.signature(target, follow_wrapped=follow_wrapped, eval_str=eval_str)
    else:
        sig = inspect.signature(target, follow_wrapped=follow_wrapped)
    check_signature(sig)
    return sig

def arg_is_required(param):
    if False:
        while True:
            i = 10
    return param.default is inspect.Parameter.empty and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)

def required_args(target, args=(), kwargs=()):
    if False:
        for i in range(10):
            print('nop')
    'Return a set of names of required args to target that were not supplied\n    in args or kwargs.\n\n    This is used in builds() to determine which arguments to attempt to\n    fill from type hints.  target may be any callable (including classes\n    and bound methods).  args and kwargs should be as they are passed to\n    builds() - that is, a tuple of values and a dict of names: values.\n    '
    if inspect.isclass(target) and is_typed_named_tuple(target):
        provided = set(kwargs) | set(target._fields[:len(args)])
        return set(target._fields) - provided
    try:
        sig = get_signature(target)
    except (ValueError, TypeError):
        return set()
    return {name for (name, param) in list(sig.parameters.items())[len(args):] if arg_is_required(param) and name not in kwargs}

def convert_keyword_arguments(function, args, kwargs):
    if False:
        while True:
            i = 10
    'Returns a pair of a tuple and a dictionary which would be equivalent\n    passed as positional and keyword args to the function. Unless function has\n    kwonlyargs or **kwargs the dictionary will always be empty.\n    '
    sig = inspect.signature(function, follow_wrapped=False)
    bound = sig.bind(*args, **kwargs)
    return (bound.args, bound.kwargs)

def convert_positional_arguments(function, args, kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Return a tuple (new_args, new_kwargs) where all possible arguments have\n    been moved to kwargs.\n\n    new_args will only be non-empty if function has pos-only args or *args.\n    '
    sig = inspect.signature(function, follow_wrapped=False)
    bound = sig.bind(*args, **kwargs)
    new_args = []
    new_kwargs = dict(bound.arguments)
    for p in sig.parameters.values():
        if p.name in new_kwargs:
            if p.kind is p.POSITIONAL_ONLY:
                new_args.append(new_kwargs.pop(p.name))
            elif p.kind is p.VAR_POSITIONAL:
                new_args.extend(new_kwargs.pop(p.name))
            elif p.kind is p.VAR_KEYWORD:
                assert set(new_kwargs[p.name]).isdisjoint(set(new_kwargs) - {p.name})
                new_kwargs.update(new_kwargs.pop(p.name))
    return (tuple(new_args), new_kwargs)

def ast_arguments_matches_signature(args, sig):
    if False:
        return 10
    assert isinstance(args, ast.arguments)
    assert isinstance(sig, inspect.Signature)
    expected = []
    for node in getattr(args, 'posonlyargs', ()):
        expected.append((node.arg, inspect.Parameter.POSITIONAL_ONLY))
    for node in args.args:
        expected.append((node.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD))
    if args.vararg is not None:
        expected.append((args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for node in args.kwonlyargs:
        expected.append((node.arg, inspect.Parameter.KEYWORD_ONLY))
    if args.kwarg is not None:
        expected.append((args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return expected == [(p.name, p.kind) for p in sig.parameters.values()]

def is_first_param_referenced_in_function(f):
    if False:
        return 10
    'Is the given name referenced within f?'
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    except Exception:
        return True
    name = next(iter(get_signature(f).parameters))
    return any((isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load) for node in ast.walk(tree)))

def extract_all_lambdas(tree, matching_signature):
    if False:
        i = 10
        return i + 15
    lambdas = []

    class Visitor(ast.NodeVisitor):

        def visit_Lambda(self, node):
            if False:
                i = 10
                return i + 15
            if ast_arguments_matches_signature(node.args, matching_signature):
                lambdas.append(node)
    Visitor().visit(tree)
    return lambdas
LINE_CONTINUATION = re.compile('\\\\\\n')
WHITESPACE = re.compile('\\s+')
PROBABLY_A_COMMENT = re.compile('#[^\'"]*$')
SPACE_FOLLOWS_OPEN_BRACKET = re.compile('\\( ')
SPACE_PRECEDES_CLOSE_BRACKET = re.compile(' \\)')

def extract_lambda_source(f):
    if False:
        return 10
    'Extracts a single lambda expression from the string source. Returns a\n    string indicating an unknown body if it gets confused in any way.\n\n    This is not a good function and I am sorry for it. Forgive me my\n    sins, oh lord\n    '
    sig = inspect.signature(f)
    assert sig.return_annotation is inspect.Parameter.empty
    if sig.parameters:
        if_confused = f'lambda {str(sig)[1:-1]}: <unknown>'
    else:
        if_confused = 'lambda: <unknown>'
    try:
        source = inspect.getsource(f)
    except OSError:
        return if_confused
    source = LINE_CONTINUATION.sub(' ', source)
    source = WHITESPACE.sub(' ', source)
    source = source.strip()
    if 'lambda' not in source and sys.platform == 'emscripten':
        return if_confused
    assert 'lambda' in source
    tree = None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        for i in range(len(source) - 1, len('lambda'), -1):
            prefix = source[:i]
            if 'lambda' not in prefix:
                break
            try:
                tree = ast.parse(prefix)
                source = prefix
                break
            except SyntaxError:
                continue
    if tree is None and source.startswith(('@', '.')):
        for i in range(len(source) + 1):
            p = source[1:i]
            if 'lambda' in p:
                try:
                    tree = ast.parse(p)
                    source = p
                    break
                except SyntaxError:
                    pass
        else:
            raise NotImplementedError('expected to be unreachable')
    if tree is None:
        return if_confused
    aligned_lambdas = extract_all_lambdas(tree, matching_signature=sig)
    if len(aligned_lambdas) != 1:
        return if_confused
    lambda_ast = aligned_lambdas[0]
    assert lambda_ast.lineno == 1
    try:
        with open(inspect.getsourcefile(f), 'rb') as src_f:
            (encoding, _) = detect_encoding(src_f.readline)
        source_bytes = source.encode(encoding)
        source_bytes = source_bytes[lambda_ast.col_offset:].strip()
        source = source_bytes.decode(encoding)
    except (OSError, TypeError):
        source = source[lambda_ast.col_offset:].strip()
    try:
        source = source[source.index('lambda'):]
    except ValueError:
        return if_confused
    for i in range(len(source), len('lambda'), -1):
        try:
            parsed = ast.parse(source[:i])
            assert len(parsed.body) == 1
            assert parsed.body
            if isinstance(parsed.body[0].value, ast.Lambda):
                source = source[:i]
                break
        except SyntaxError:
            pass
    lines = source.split('\n')
    lines = [PROBABLY_A_COMMENT.sub('', l) for l in lines]
    source = '\n'.join(lines)
    source = WHITESPACE.sub(' ', source)
    source = SPACE_FOLLOWS_OPEN_BRACKET.sub('(', source)
    source = SPACE_PRECEDES_CLOSE_BRACKET.sub(')', source)
    return source.strip()

def get_pretty_function_description(f):
    if False:
        print('Hello World!')
    if not hasattr(f, '__name__'):
        return repr(f)
    name = f.__name__
    if name == '<lambda>':
        return extract_lambda_source(f)
    elif isinstance(f, (types.MethodType, types.BuiltinMethodType)):
        self = f.__self__
        if not (self is None or inspect.isclass(self) or inspect.ismodule(self)):
            return f'{self!r}.{name}'
    elif isinstance(name, str) and getattr(dict, name, object()) is f:
        return f'dict.{name}'
    return name

def nicerepr(v):
    if False:
        i = 10
        return i + 15
    if inspect.isfunction(v):
        return get_pretty_function_description(v)
    elif isinstance(v, type):
        return v.__name__
    else:
        return re.sub('(\\[)~([A-Z][a-z]*\\])', '\\g<1>\\g<2>', pretty(v))

def repr_call(f, args, kwargs, *, reorder=True):
    if False:
        while True:
            i = 10
    if reorder:
        (args, kwargs) = convert_positional_arguments(f, args, kwargs)
    bits = [nicerepr(x) for x in args]
    for p in get_signature(f).parameters.values():
        if p.name in kwargs and (not p.kind.name.startswith('VAR_')):
            bits.append(f'{p.name}={nicerepr(kwargs.pop(p.name))}')
    if kwargs:
        for a in sorted(kwargs):
            bits.append(f'{a}={nicerepr(kwargs[a])}')
    rep = nicerepr(f)
    if rep.startswith('lambda') and ':' in rep:
        rep = f'({rep})'
    return rep + '(' + ', '.join(bits) + ')'

def check_valid_identifier(identifier):
    if False:
        for i in range(10):
            print('nop')
    if not identifier.isidentifier():
        raise ValueError(f'{identifier!r} is not a valid python identifier')
eval_cache: dict = {}

def source_exec_as_module(source):
    if False:
        return 10
    try:
        return eval_cache[source]
    except KeyError:
        pass
    hexdigest = hashlib.sha384(source.encode()).hexdigest()
    result = ModuleType('hypothesis_temporary_module_' + hexdigest)
    assert isinstance(source, str)
    exec(source, result.__dict__)
    eval_cache[source] = result
    return result
COPY_SIGNATURE_SCRIPT = '\nfrom hypothesis.utils.conventions import not_set\n\ndef accept({funcname}):\n    def {name}{signature}:\n        return {funcname}({invocation})\n    return {name}\n'.lstrip()

def get_varargs(sig, kind=inspect.Parameter.VAR_POSITIONAL):
    if False:
        return 10
    for p in sig.parameters.values():
        if p.kind is kind:
            return p
    return None

def define_function_signature(name, docstring, signature):
    if False:
        return 10
    'A decorator which sets the name, signature and docstring of the function\n    passed into it.'
    if name == '<lambda>':
        name = '_lambda_'
    check_valid_identifier(name)
    for a in signature.parameters:
        check_valid_identifier(a)
    used_names = {*signature.parameters, name}
    newsig = signature.replace(parameters=[p if p.default is signature.empty else p.replace(default=not_set) for p in (p.replace(annotation=signature.empty) for p in signature.parameters.values())], return_annotation=signature.empty)
    pos_args = [p for p in signature.parameters.values() if p.kind.name.startswith('POSITIONAL_')]

    def accept(f):
        if False:
            return 10
        fsig = inspect.signature(f, follow_wrapped=False)
        must_pass_as_kwargs = []
        invocation_parts = []
        for p in pos_args:
            if p.name not in fsig.parameters and get_varargs(fsig) is None:
                must_pass_as_kwargs.append(p.name)
            else:
                invocation_parts.append(p.name)
        if get_varargs(signature) is not None:
            invocation_parts.append('*' + get_varargs(signature).name)
        for k in must_pass_as_kwargs:
            invocation_parts.append(f'{k}={k}')
        for p in signature.parameters.values():
            if p.kind is p.KEYWORD_ONLY:
                invocation_parts.append(f'{p.name}={p.name}')
        varkw = get_varargs(signature, kind=inspect.Parameter.VAR_KEYWORD)
        if varkw:
            invocation_parts.append('**' + varkw.name)
        candidate_names = ['f'] + [f'f_{i}' for i in range(1, len(used_names) + 2)]
        for funcname in candidate_names:
            if funcname not in used_names:
                break
        source = COPY_SIGNATURE_SCRIPT.format(name=name, funcname=funcname, signature=str(newsig), invocation=', '.join(invocation_parts))
        result = source_exec_as_module(source).accept(f)
        result.__doc__ = docstring
        result.__defaults__ = tuple((p.default for p in signature.parameters.values() if p.default is not signature.empty and 'POSITIONAL' in p.kind.name))
        kwdefaults = {p.name: p.default for p in signature.parameters.values() if p.default is not signature.empty and p.kind is p.KEYWORD_ONLY}
        if kwdefaults:
            result.__kwdefaults__ = kwdefaults
        annotations = {p.name: p.annotation for p in signature.parameters.values() if p.annotation is not signature.empty}
        if signature.return_annotation is not signature.empty:
            annotations['return'] = signature.return_annotation
        if annotations:
            result.__annotations__ = annotations
        return result
    return accept

def impersonate(target):
    if False:
        i = 10
        return i + 15
    "Decorator to update the attributes of a function so that to external\n    introspectors it will appear to be the target function.\n\n    Note that this updates the function in place, it doesn't return a\n    new one.\n    "

    def accept(f):
        if False:
            return 10
        f.__code__ = f.__code__.replace(co_filename=target.__code__.co_filename, co_firstlineno=target.__code__.co_firstlineno)
        f.__name__ = target.__name__
        f.__module__ = target.__module__
        f.__doc__ = target.__doc__
        f.__globals__['__hypothesistracebackhide__'] = True
        return f
    return accept

def proxies(target: 'T') -> Callable[[Callable], 'T']:
    if False:
        i = 10
        return i + 15
    replace_sig = define_function_signature(target.__name__.replace('<lambda>', '_lambda_'), target.__doc__, get_signature(target, follow_wrapped=False))

    def accept(proxy):
        if False:
            i = 10
            return i + 15
        return impersonate(target)(wraps(target)(replace_sig(proxy)))
    return accept

def is_identity_function(f):
    if False:
        print('Hello World!')
    return bool(re.fullmatch('lambda (\\w+): \\1', get_pretty_function_description(f)))