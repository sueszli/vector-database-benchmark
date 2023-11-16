"""
    Command-line tool to expose qtile.command functionality to shell.
    This can be used standalone or in other shell scripts.
"""
from __future__ import annotations
import argparse
import itertools
import pprint
import sys
import textwrap
from libqtile.command.base import CommandError, CommandException, SelectError
from libqtile.command.client import CommandClient
from libqtile.command.graph import CommandGraphRoot
from libqtile.command.interface import IPCCommandInterface
from libqtile.ipc import Client, find_sockfile

def get_formated_info(obj: CommandClient, cmd: str, args=True, short=True) -> str:
    if False:
        i = 10
        return i + 15
    "Get documentation for command/function and format it.\n\n    Returns:\n      * args=True, short=True - '*' if arguments are present and a summary line.\n      * args=True, short=False - (function args) and a summary line.\n      * args=False - a summary line.\n\n    If 'doc' function is not present in object or there is no doc string for\n    given cmd it returns empty string.  The arguments are extracted from doc[0]\n    line, the summary is constructed from doc[1] line.\n    "
    doc = obj.call('doc', cmd).splitlines()
    tdoc = doc[0]
    doc_args = tdoc[tdoc.find('('):tdoc.find(')') + 1].strip()
    short_description = doc[1] if len(doc) > 1 else ''
    if not args:
        doc_args = ''
    elif short:
        doc_args = ' ' if doc_args == '()' else '*'
    return (doc_args + ' ' + short_description).rstrip()

def print_commands(prefix: str, obj: CommandClient) -> None:
    if False:
        i = 10
        return i + 15
    'Print available commands for given object.'
    prefix += ' -f '
    cmds = obj.call('commands')
    output = []
    for cmd in cmds:
        doc_args = get_formated_info(obj, cmd)
        pcmd = prefix + cmd
        output.append([pcmd, doc_args])
    max_cmd = max((len(pcmd) for (pcmd, _) in output))
    formatting = '{:<%d}\t{}' % (max_cmd + 1)
    for line in output:
        print(formatting.format(line[0], line[1]))

def get_object(client: CommandClient, argv: list[str]) -> CommandClient:
    if False:
        while True:
            i = 10
    '\n    Constructs a path to object and returns given object (if it exists).\n    '
    if argv[0] == 'cmd':
        argv = argv[1:]
    parsed_next = False
    for (arg0, arg1) in itertools.zip_longest(argv, argv[1:]):
        if parsed_next:
            parsed_next = False
            continue
        try:
            client = client.navigate(arg0, arg1)
            parsed_next = True
            continue
        except SelectError:
            pass
        try:
            client = client.navigate(arg0, None)
            continue
        except SelectError:
            pass
        print('Specified object does not exist: ' + ' '.join(argv))
        sys.exit(1)
    return client

def run_function(client: CommandClient, funcname: str, args: list[str]) -> str:
    if False:
        return 10
    'Run command with specified args on given object.'
    try:
        ret = client.call(funcname, *args)
    except SelectError:
        print('error: Sorry no function ', funcname)
        sys.exit(1)
    except CommandError as e:
        print("error: Command '{}' returned error: {}".format(funcname, str(e)))
        sys.exit(1)
    except CommandException as e:
        print("error: Sorry cannot run function '{}' with arguments {}: {}".format(funcname, args, str(e)))
        sys.exit(1)
    return ret

def print_base_objects() -> None:
    if False:
        return 10
    'Prints access objects of Client, use cmd for commands.'
    root = CommandGraphRoot()
    actions = ['-o cmd'] + [f'-o {key}' for key in root.children]
    print('Specify an object on which to execute command')
    print('\n'.join(actions))

def cmd_obj(args) -> None:
    if False:
        return 10
    'Runs tool according to specified arguments.'
    if args.obj_spec:
        sock_file = args.socket or find_sockfile()
        ipc_client = Client(sock_file)
        cmd_object = IPCCommandInterface(ipc_client)
        cmd_client = CommandClient(cmd_object)
        obj = get_object(cmd_client, args.obj_spec)
        if args.function == 'help':
            try:
                print_commands('-o ' + ' '.join(args.obj_spec), obj)
            except CommandError:
                if len(args.obj_spec) == 1:
                    print(f"{args.obj_spec} object needs a specified identifier e.g. '-o bar top'.")
                    sys.exit(1)
                else:
                    raise
        elif args.info:
            print(args.function + get_formated_info(obj, args.function, args=True, short=False))
        else:
            ret = run_function(obj, args.function, args.args)
            if ret is not None:
                pprint.pprint(ret)
    else:
        print_base_objects()
        sys.exit(1)

def add_subcommand(subparsers, parents):
    if False:
        print('Hello World!')
    epilog = textwrap.dedent('\n        Examples:\n         qtile cmd-obj\n         qtile cmd-obj -o cmd\n         qtile cmd-obj -o cmd -f prev_layout -i\n         qtile cmd-obj -o cmd -f prev_layout -a 3 # prev_layout on group 3\n         qtile cmd-obj -o group 3 -f focus_back\n         qtile cmd-obj -o cmd -f restart # restart qtile\n        ')
    description = 'Access the command interface from a shell.'
    parser = subparsers.add_parser('cmd-obj', help=description, parents=parents, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--object', '-o', dest='obj_spec', nargs='+', help='Specify path to object (space separated).  If no --function flag display available commands.  Use `cmd` to specify root command.')
    parser.add_argument('--function', '-f', default='help', help='Select function to execute.')
    parser.add_argument('--args', '-a', nargs='+', default=[], help='Set arguments supplied to function.')
    parser.add_argument('--info', '-i', action='store_true', help='With both --object and --function args prints documentation for function.')
    parser.add_argument('--socket', '-s', help='Use specified socket for IPC.')
    parser.set_defaults(func=cmd_obj)