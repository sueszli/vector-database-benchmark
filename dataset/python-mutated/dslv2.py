"""
The DSL is used for RPC and resource definition. The syntax of the DSL is same
as UNIX shell.

>>> req = parse("search jaychou -s=xx")
>>> req.cmd
'search'
>>> req.cmd_args
['jaychou']
"""
import argparse
import shlex
import itertools
from typing import Optional
from feeluown.argparser import create_fmt_parser, add_common_cmds, add_server_cmds
from feeluown.server.data_structure import Request
from feeluown.server.excs import FuoSyntaxError

def tokenize(source):
    if False:
        i = 10
        return i + 15
    s = shlex.shlex(source, punctuation_chars=True, posix=True)
    s.whitespace_split = True
    try:
        tokens = list(s)
    except ValueError as e:
        raise FuoSyntaxError(str(e)) from None
    else:
        if s.lineno >= 2:
            raise FuoSyntaxError('source must be only one line')
        return tokens

class ArgumentParserNoExitAndPrint(argparse.ArgumentParser):
    """
    This customized argument parser class is design to handle two scenario
    1. When there is an error, the parser should not exit. So the error method is
       overrided.
    2. When `help` action is executed, the parser should not exit and it should
       not print help message to stdout/stderr either. So the `_print_message` and
       `exit` method are overrided.
    """

    def _print_message(self, message, file=None):
        if False:
            return 10
        pass

    def exit(self, status=0, message=None):
        if False:
            print('Hello World!')
        pass

    def error(self, message):
        if False:
            for i in range(10):
                print('nop')
        raise FuoSyntaxError(message)

def create_dsl_parser():
    if False:
        i = 10
        return i + 15
    parser = ArgumentParserNoExitAndPrint(add_help=False)
    subparsers = parser.add_subparsers(dest='cmd')
    add_common_cmds(subparsers)
    add_server_cmds(subparsers)
    return parser

class Parser:

    def __init__(self, source):
        if False:
            i = 10
            return i + 15
        self._source = source

    def parse(self) -> Request:
        if False:
            print('Hello World!')
        'Parse the source to a Request object.\n\n        argparse have little public methods, so some protected methods are used.\n        '
        parser: ArgumentParserNoExitAndPrint = create_dsl_parser()
        tokens = tokenize(self._source)
        (has_heredoc, heredoc_word) = (False, None)
        for (i, token) in enumerate(tokens.copy()):
            if token == '<<':
                has_heredoc = True
                try:
                    heredoc_word = tokens.pop(i + 1)
                except IndexError:
                    raise FuoSyntaxError('no heredoc word') from None
                else:
                    tokens.pop(i)
            elif token in ('<', '<<<'):
                raise FuoSyntaxError('unknown token')
        (args, remain) = parser.parse_known_args(tokens)
        if remain:
            raise FuoSyntaxError(f'unknown tokens {tokens}')
        cmdname = getattr(args, 'cmd')
        subparser = get_subparser(parser, cmdname)
        assert subparser is not None, f'parser for cmd:{cmdname} not found'
        cmd_args = []
        for action in subparser._positionals._group_actions:
            cmd_args.append(getattr(args, action.dest))
        req_options = {}
        option_names_req = []
        for parser_ in [create_fmt_parser()]:
            for action in parser_._actions:
                name = action.dest
                option_names_req.append(name)
                value = getattr(args, name)
                req_options[name] = value
        cmd_options = {}
        for action in subparser._optionals._group_actions:
            option_name = action.dest
            if option_name == 'help':
                continue
            if option_name not in option_names_req:
                cmd_options[option_name] = getattr(args, option_name)
        return Request(cmdname, cmd_args, cmd_options, req_options, has_heredoc=has_heredoc, heredoc_word=heredoc_word)

def get_subparser(parser, cmdname) -> Optional[argparse.ArgumentParser]:
    if False:
        i = 10
        return i + 15
    root_dest = 'cmd'
    subparser = None
    for action in parser._actions:
        if action.dest == root_dest:
            subparser = action._name_parser_map[cmdname]
            break
    return subparser

def parse(source):
    if False:
        for i in range(10):
            print('nop')
    return Parser(source).parse()

def unparse(request: Request):
    if False:
        while True:
            i = 10
    'Generate source code for the request object'
    parser = create_dsl_parser()
    subparser = get_subparser(parser, request.cmd)
    if subparser is None:
        raise ValueError(f'{request.cmd}: no such cmd')
    cmdline = [request.cmd]
    if request.has_heredoc:
        cmdline.append(f'<<{request.heredoc_word}')
    else:
        cmdline.extend([shlex.quote(each) for each in request.cmd_args])
    for (key, value) in itertools.chain(request.cmd_options.items(), request.options.items()):
        for action in subparser._actions:
            if action.dest == key:
                if value is None:
                    break
                if isinstance(action, argparse._StoreTrueAction):
                    if value is True:
                        cmdline.append(f'--{key}')
                elif isinstance(action, argparse._AppendAction):
                    for each in value or []:
                        cmdline.append(f'--{key}={shlex.quote(str(each))}')
                else:
                    cmdline.append(f'--{key}={shlex.quote(str(value))}')
                break
        else:
            raise ValueError(f'{key}: no such option')
    cmdtext = ' '.join(cmdline)
    if request.has_heredoc:
        cmdtext += '\n'
        cmdtext += request.cmd_args[0]
        cmdtext += '\n'
        cmdtext += request.heredoc_word
    return cmdtext