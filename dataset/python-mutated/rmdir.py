from xonsh.completers.path import complete_dir
from xonsh.parsers.completion_context import CommandContext

def xonsh_complete(ctx: CommandContext):
    if False:
        print('Hello World!')
    '\n    Completion for "rmdir", includes only valid directory names.\n    '
    if not ctx.prefix.startswith('-') and ctx.arg_index > 0:
        (comps, lprefix) = complete_dir(ctx)
        if not comps:
            raise StopIteration
        return (comps, lprefix)