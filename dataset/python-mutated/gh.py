"""Completers for gh CLI"""
from xonsh.completers.tools import completion_from_cmd_output, sub_proc_get_output
from xonsh.parsers.completion_context import CommandContext

def _complete(cmd, *args):
    if False:
        while True:
            i = 10
    (out, _) = sub_proc_get_output(cmd, '__complete', *args)
    if out:
        (*lines, dir_num) = out.decode().splitlines()
        for ln in lines:
            yield completion_from_cmd_output(ln)

def xonsh_complete(ctx: CommandContext):
    if False:
        print('Hello World!')
    (cmd, *args) = [arg.value for arg in ctx.args] + [ctx.prefix]
    return _complete(cmd, *args)