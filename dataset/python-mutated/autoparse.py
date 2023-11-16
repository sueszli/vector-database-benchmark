import sys
from re import compile as compile_regex
from inspect import signature, getdoc, Parameter
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import wraps
from io import IOBase
from autocommand.errors import AutocommandError
_empty = Parameter.empty

class AnnotationError(AutocommandError):
    """Annotation error: annotation must be a string, type, or tuple of both"""

class PositionalArgError(AutocommandError):
    """
    Postional Arg Error: autocommand can't handle postional-only parameters
    """

class KWArgError(AutocommandError):
    """kwarg Error: autocommand can't handle a **kwargs parameter"""

class DocstringError(AutocommandError):
    """Docstring error"""

class TooManySplitsError(DocstringError):
    """
    The docstring had too many ---- section splits. Currently we only support
    using up to a single split, to split the docstring into description and
    epilog parts.
    """

def _get_type_description(annotation):
    if False:
        while True:
            i = 10
    '\n    Given an annotation, return the (type, description) for the parameter.\n    If you provide an annotation that is somehow both a string and a callable,\n    the behavior is undefined.\n    '
    if annotation is _empty:
        return (None, None)
    elif callable(annotation):
        return (annotation, None)
    elif isinstance(annotation, str):
        return (None, annotation)
    elif isinstance(annotation, tuple):
        try:
            (arg1, arg2) = annotation
        except ValueError as e:
            raise AnnotationError(annotation) from e
        else:
            if callable(arg1) and isinstance(arg2, str):
                return (arg1, arg2)
            elif isinstance(arg1, str) and callable(arg2):
                return (arg2, arg1)
    raise AnnotationError(annotation)

def _add_arguments(param, parser, used_char_args, add_nos):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add the argument(s) to an ArgumentParser (using add_argument) for a given\n    parameter. used_char_args is the set of -short options currently already in\n    use, and is updated (if necessary) by this function. If add_nos is True,\n    this will also add an inverse switch for all boolean options. For\n    instance, for the boolean parameter "verbose", this will create --verbose\n    and --no-verbose.\n    '
    if param.kind is param.POSITIONAL_ONLY:
        raise PositionalArgError(param)
    elif param.kind is param.VAR_KEYWORD:
        raise KWArgError(param)
    arg_spec = {}
    is_option = False
    (arg_type, description) = _get_type_description(param.annotation)
    default = param.default
    if arg_type is None and default not in {_empty, None}:
        arg_type = type(default)
    if default is not _empty:
        arg_spec['default'] = default
        is_option = True
    if arg_type is not None:
        if arg_type is bool:
            if not default or default is _empty:
                arg_spec['action'] = 'store_true'
            else:
                arg_spec['action'] = 'store_false'
            is_option = True
        elif isinstance(default, IOBase):
            arg_spec['type'] = str
        else:
            arg_spec['type'] = arg_type
    if param.kind is param.VAR_POSITIONAL:
        arg_spec['nargs'] = '*'
    if description is not None:
        arg_spec['help'] = description
    flags = []
    name = param.name
    if is_option:
        for letter in (name[0], name[0].swapcase()):
            if letter not in used_char_args:
                used_char_args.add(letter)
                flags.append('-{}'.format(letter))
                break
        if len(name) > 1 or not flags:
            flags.append('--{}'.format(name))
        arg_spec['dest'] = name
    else:
        flags.append(name)
    parser.add_argument(*flags, **arg_spec)
    if add_nos and arg_type is bool:
        parser.add_argument('--no-{}'.format(name), action='store_const', dest=name, const=default if default is not _empty else False)

def make_parser(func_sig, description, epilog, add_nos):
    if False:
        print('Hello World!')
    '\n    Given the signature of a function, create an ArgumentParser\n    '
    parser = ArgumentParser(description=description, epilog=epilog)
    used_char_args = {'h'}
    params = sorted(func_sig.parameters.values(), key=lambda param: len(param.name) > 1)
    for param in params:
        _add_arguments(param, parser, used_char_args, add_nos)
    return parser
_DOCSTRING_SPLIT = compile_regex('\\n\\s*-{4,}\\s*\\n')

def parse_docstring(docstring):
    if False:
        i = 10
        return i + 15
    '\n    Given a docstring, parse it into a description and epilog part\n    '
    if docstring is None:
        return ('', '')
    parts = _DOCSTRING_SPLIT.split(docstring)
    if len(parts) == 1:
        return (docstring, '')
    elif len(parts) == 2:
        return (parts[0], parts[1])
    else:
        raise TooManySplitsError()

def autoparse(func=None, *, description=None, epilog=None, add_nos=False, parser=None):
    if False:
        while True:
            i = 10
    "\n    This decorator converts a function that takes normal arguments into a\n    function which takes a single optional argument, argv, parses it using an\n    argparse.ArgumentParser, and calls the underlying function with the parsed\n    arguments. If it is not given, sys.argv[1:] is used. This is so that the\n    function can be used as a setuptools entry point, as well as a normal main\n    function. sys.argv[1:] is not evaluated until the function is called, to\n    allow injecting different arguments for testing.\n\n    It uses the argument signature of the function to create an\n    ArgumentParser. Parameters without defaults become positional parameters,\n    while parameters *with* defaults become --options. Use annotations to set\n    the type of the parameter.\n\n    The `desctiption` and `epilog` parameters corrospond to the same respective\n    argparse parameters. If no description is given, it defaults to the\n    decorated functions's docstring, if present.\n\n    If add_nos is True, every boolean option (that is, every parameter with a\n    default of True/False or a type of bool) will have a --no- version created\n    as well, which inverts the option. For instance, the --verbose option will\n    have a --no-verbose counterpart. These are not mutually exclusive-\n    whichever one appears last in the argument list will have precedence.\n\n    If a parser is given, it is used instead of one generated from the function\n    signature. In this case, no parser is created; instead, the given parser is\n    used to parse the argv argument. The parser's results' argument names must\n    match up with the parameter names of the decorated function.\n\n    The decorated function is attached to the result as the `func` attribute,\n    and the parser is attached as the `parser` attribute.\n    "
    if func is None:
        return lambda f: autoparse(f, description=description, epilog=epilog, add_nos=add_nos, parser=parser)
    func_sig = signature(func)
    (docstr_description, docstr_epilog) = parse_docstring(getdoc(func))
    if parser is None:
        parser = make_parser(func_sig, description or docstr_description, epilog or docstr_epilog, add_nos)

    @wraps(func)
    def autoparse_wrapper(argv=None):
        if False:
            i = 10
            return i + 15
        if argv is None:
            argv = sys.argv[1:]
        parsed_args = func_sig.bind_partial()
        parsed_args.arguments.update(vars(parser.parse_args(argv)))
        return func(*parsed_args.args, **parsed_args.kwargs)
    autoparse_wrapper.func = func
    autoparse_wrapper.parser = parser
    return autoparse_wrapper

@contextmanager
def smart_open(filename_or_file, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    This context manager allows you to open a filename, if you want to default\n    some already-existing file object, like sys.stdout, which shouldn\'t be\n    closed at the end of the context. If the filename argument is a str, bytes,\n    or int, the file object is created via a call to open with the given *args\n    and **kwargs, sent to the context, and closed at the end of the context,\n    just like "with open(filename) as f:". If it isn\'t one of the openable\n    types, the object simply sent to the context unchanged, and left unclosed\n    at the end of the context. Example:\n\n        def work_with_file(name=sys.stdout):\n            with smart_open(name) as f:\n                # Works correctly if name is a str filename or sys.stdout\n                print("Some stuff", file=f)\n                # If it was a filename, f is closed at the end here.\n    '
    if isinstance(filename_or_file, (str, bytes, int)):
        with open(filename_or_file, *args, **kwargs) as file:
            yield file
    else:
        yield filename_or_file