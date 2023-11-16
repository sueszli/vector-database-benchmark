"""
    Command-line wrapper to run commands and add rules to new windows
"""
import argparse
import atexit
import subprocess
from libqtile import ipc
from libqtile.command import graph

def run_cmd(opts) -> None:
    if False:
        i = 10
        return i + 15
    if opts.socket is None:
        socket = ipc.find_sockfile()
    else:
        socket = opts.socket
    client = ipc.Client(socket)
    root = graph.CommandGraphRoot()
    cmd = [opts.cmd]
    if opts.args:
        cmd.extend(opts.args)
    proc = subprocess.Popen(cmd)
    match_args = {'net_wm_pid': proc.pid}
    rule_args = {'float': opts.float, 'intrusive': opts.intrusive, 'group': opts.group, 'break_on_match': not opts.dont_break}
    graph_cmd = root.call('add_rule')
    (_, rule_id) = client.send((root.selectors, graph_cmd.name, (match_args, rule_args), {}))

    def remove_rule() -> None:
        if False:
            print('Hello World!')
        cmd = root.call('remove_rule')
        client.send((root.selectors, cmd.name, (rule_id,), {}))
    atexit.register(remove_rule)
    proc.wait()

def add_subcommand(subparsers, parents):
    if False:
        for i in range(10):
            print('nop')
    parser = subparsers.add_parser('run-cmd', parents=parents, help='A wrapper around the command graph.')
    parser.add_argument('-s', '--socket', help='Use specified socket for IPC.')
    parser.add_argument('-i', '--intrusive', action='store_true', help='If the new window should be intrusive.')
    parser.add_argument('-f', '--float', action='store_true', help='If the new window should be floating.')
    parser.add_argument('-b', '--dont-break', action='store_true', help='Do not break on match (keep applying rules).')
    parser.add_argument('-g', '--group', help='Set the window group.')
    (parser.add_argument('cmd', help='Command to execute.'),)
    parser.add_argument('args', nargs=argparse.REMAINDER, metavar='[args ...]', help='Optional arguments to pass to command.')
    parser.set_defaults(func=run_cmd)