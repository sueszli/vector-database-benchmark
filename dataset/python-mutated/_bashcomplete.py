import copy
import os
import re
from .core import Argument
from .core import MultiCommand
from .core import Option
from .parser import split_arg_string
from .types import Choice
from .utils import echo
try:
    from collections import abc
except ImportError:
    import collections as abc
WORDBREAK = '='
COMPLETION_SCRIPT_BASH = '\n%(complete_func)s() {\n    local IFS=$\'\n\'\n    COMPREPLY=( $( env COMP_WORDS="${COMP_WORDS[*]}" \\\n                   COMP_CWORD=$COMP_CWORD \\\n                   %(autocomplete_var)s=complete $1 ) )\n    return 0\n}\n\n%(complete_func)setup() {\n    local COMPLETION_OPTIONS=""\n    local BASH_VERSION_ARR=(${BASH_VERSION//./ })\n    # Only BASH version 4.4 and later have the nosort option.\n    if [ ${BASH_VERSION_ARR[0]} -gt 4 ] || ([ ${BASH_VERSION_ARR[0]} -eq 4 ] && [ ${BASH_VERSION_ARR[1]} -ge 4 ]); then\n        COMPLETION_OPTIONS="-o nosort"\n    fi\n\n    complete $COMPLETION_OPTIONS -F %(complete_func)s %(script_names)s\n}\n\n%(complete_func)setup\n'
COMPLETION_SCRIPT_ZSH = '\n#compdef %(script_names)s\n\n%(complete_func)s() {\n    local -a completions\n    local -a completions_with_descriptions\n    local -a response\n    (( ! $+commands[%(script_names)s] )) && return 1\n\n    response=("${(@f)$( env COMP_WORDS="${words[*]}" \\\n                        COMP_CWORD=$((CURRENT-1)) \\\n                        %(autocomplete_var)s="complete_zsh" \\\n                        %(script_names)s )}")\n\n    for key descr in ${(kv)response}; do\n      if [[ "$descr" == "_" ]]; then\n          completions+=("$key")\n      else\n          completions_with_descriptions+=("$key":"$descr")\n      fi\n    done\n\n    if [ -n "$completions_with_descriptions" ]; then\n        _describe -V unsorted completions_with_descriptions -U\n    fi\n\n    if [ -n "$completions" ]; then\n        compadd -U -V unsorted -a completions\n    fi\n    compstate[insert]="automenu"\n}\n\ncompdef %(complete_func)s %(script_names)s\n'
COMPLETION_SCRIPT_FISH = 'complete --no-files --command %(script_names)s --arguments "(env %(autocomplete_var)s=complete_fish COMP_WORDS=(commandline -cp) COMP_CWORD=(commandline -t) %(script_names)s)"'
_completion_scripts = {'bash': COMPLETION_SCRIPT_BASH, 'zsh': COMPLETION_SCRIPT_ZSH, 'fish': COMPLETION_SCRIPT_FISH}
_invalid_ident_char_re = re.compile('[^a-zA-Z0-9_]')

def get_completion_script(prog_name, complete_var, shell):
    if False:
        return 10
    cf_name = _invalid_ident_char_re.sub('', prog_name.replace('-', '_'))
    script = _completion_scripts.get(shell, COMPLETION_SCRIPT_BASH)
    return (script % {'complete_func': '_{}_completion'.format(cf_name), 'script_names': prog_name, 'autocomplete_var': complete_var}).strip() + ';'

def resolve_ctx(cli, prog_name, args):
    if False:
        i = 10
        return i + 15
    'Parse into a hierarchy of contexts. Contexts are connected\n    through the parent variable.\n\n    :param cli: command definition\n    :param prog_name: the program that is running\n    :param args: full list of args\n    :return: the final context/command parsed\n    '
    ctx = cli.make_context(prog_name, args, resilient_parsing=True)
    args = ctx.protected_args + ctx.args
    while args:
        if isinstance(ctx.command, MultiCommand):
            if not ctx.command.chain:
                (cmd_name, cmd, args) = ctx.command.resolve_command(ctx, args)
                if cmd is None:
                    return ctx
                ctx = cmd.make_context(cmd_name, args, parent=ctx, resilient_parsing=True)
                args = ctx.protected_args + ctx.args
            else:
                while args:
                    (cmd_name, cmd, args) = ctx.command.resolve_command(ctx, args)
                    if cmd is None:
                        return ctx
                    sub_ctx = cmd.make_context(cmd_name, args, parent=ctx, allow_extra_args=True, allow_interspersed_args=False, resilient_parsing=True)
                    args = sub_ctx.args
                ctx = sub_ctx
                args = sub_ctx.protected_args + sub_ctx.args
        else:
            break
    return ctx

def start_of_option(param_str):
    if False:
        i = 10
        return i + 15
    '\n    :param param_str: param_str to check\n    :return: whether or not this is the start of an option declaration\n        (i.e. starts "-" or "--")\n    '
    return param_str and param_str[:1] == '-'

def is_incomplete_option(all_args, cmd_param):
    if False:
        print('Hello World!')
    '\n    :param all_args: the full original list of args supplied\n    :param cmd_param: the current command paramter\n    :return: whether or not the last option declaration (i.e. starts\n        "-" or "--") is incomplete and corresponds to this cmd_param. In\n        other words whether this cmd_param option can still accept\n        values\n    '
    if not isinstance(cmd_param, Option):
        return False
    if cmd_param.is_flag:
        return False
    last_option = None
    for (index, arg_str) in enumerate(reversed([arg for arg in all_args if arg != WORDBREAK])):
        if index + 1 > cmd_param.nargs:
            break
        if start_of_option(arg_str):
            last_option = arg_str
    return True if last_option and last_option in cmd_param.opts else False

def is_incomplete_argument(current_params, cmd_param):
    if False:
        return 10
    '\n    :param current_params: the current params and values for this\n        argument as already entered\n    :param cmd_param: the current command parameter\n    :return: whether or not the last argument is incomplete and\n        corresponds to this cmd_param. In other words whether or not the\n        this cmd_param argument can still accept values\n    '
    if not isinstance(cmd_param, Argument):
        return False
    current_param_values = current_params[cmd_param.name]
    if current_param_values is None:
        return True
    if cmd_param.nargs == -1:
        return True
    if isinstance(current_param_values, abc.Iterable) and cmd_param.nargs > 1 and (len(current_param_values) < cmd_param.nargs):
        return True
    return False

def get_user_autocompletions(ctx, args, incomplete, cmd_param):
    if False:
        while True:
            i = 10
    '\n    :param ctx: context associated with the parsed command\n    :param args: full list of args\n    :param incomplete: the incomplete text to autocomplete\n    :param cmd_param: command definition\n    :return: all the possible user-specified completions for the param\n    '
    results = []
    if isinstance(cmd_param.type, Choice):
        results = [(c, None) for c in cmd_param.type.choices if str(c).startswith(incomplete)]
    elif cmd_param.autocompletion is not None:
        dynamic_completions = cmd_param.autocompletion(ctx=ctx, args=args, incomplete=incomplete)
        results = [c if isinstance(c, tuple) else (c, None) for c in dynamic_completions]
    return results

def get_visible_commands_starting_with(ctx, starts_with):
    if False:
        i = 10
        return i + 15
    '\n    :param ctx: context associated with the parsed command\n    :starts_with: string that visible commands must start with.\n    :return: all visible (not hidden) commands that start with starts_with.\n    '
    for c in ctx.command.list_commands(ctx):
        if c.startswith(starts_with):
            command = ctx.command.get_command(ctx, c)
            if not command.hidden:
                yield command

def add_subcommand_completions(ctx, incomplete, completions_out):
    if False:
        while True:
            i = 10
    if isinstance(ctx.command, MultiCommand):
        completions_out.extend([(c.name, c.get_short_help_str()) for c in get_visible_commands_starting_with(ctx, incomplete)])
    while ctx.parent is not None:
        ctx = ctx.parent
        if isinstance(ctx.command, MultiCommand) and ctx.command.chain:
            remaining_commands = [c for c in get_visible_commands_starting_with(ctx, incomplete) if c.name not in ctx.protected_args]
            completions_out.extend([(c.name, c.get_short_help_str()) for c in remaining_commands])

def get_choices(cli, prog_name, args, incomplete):
    if False:
        while True:
            i = 10
    '\n    :param cli: command definition\n    :param prog_name: the program that is running\n    :param args: full list of args\n    :param incomplete: the incomplete text to autocomplete\n    :return: all the possible completions for the incomplete\n    '
    all_args = copy.deepcopy(args)
    ctx = resolve_ctx(cli, prog_name, args)
    if ctx is None:
        return []
    has_double_dash = '--' in all_args
    if start_of_option(incomplete) and WORDBREAK in incomplete:
        partition_incomplete = incomplete.partition(WORDBREAK)
        all_args.append(partition_incomplete[0])
        incomplete = partition_incomplete[2]
    elif incomplete == WORDBREAK:
        incomplete = ''
    completions = []
    if not has_double_dash and start_of_option(incomplete):
        for param in ctx.command.params:
            if isinstance(param, Option) and (not param.hidden):
                param_opts = [param_opt for param_opt in param.opts + param.secondary_opts if param_opt not in all_args or param.multiple]
                completions.extend([(o, param.help) for o in param_opts if o.startswith(incomplete)])
        return completions
    for param in ctx.command.params:
        if is_incomplete_option(all_args, param):
            return get_user_autocompletions(ctx, all_args, incomplete, param)
    for param in ctx.command.params:
        if is_incomplete_argument(ctx.params, param):
            return get_user_autocompletions(ctx, all_args, incomplete, param)
    add_subcommand_completions(ctx, incomplete, completions)
    return sorted(completions)

def do_complete(cli, prog_name, include_descriptions):
    if False:
        print('Hello World!')
    cwords = split_arg_string(os.environ['COMP_WORDS'])
    cword = int(os.environ['COMP_CWORD'])
    args = cwords[1:cword]
    try:
        incomplete = cwords[cword]
    except IndexError:
        incomplete = ''
    for item in get_choices(cli, prog_name, args, incomplete):
        echo(item[0])
        if include_descriptions:
            echo(item[1] if item[1] else '_')
    return True

def do_complete_fish(cli, prog_name):
    if False:
        i = 10
        return i + 15
    cwords = split_arg_string(os.environ['COMP_WORDS'])
    incomplete = os.environ['COMP_CWORD']
    args = cwords[1:]
    for item in get_choices(cli, prog_name, args, incomplete):
        if item[1]:
            echo('{arg}\t{desc}'.format(arg=item[0], desc=item[1]))
        else:
            echo(item[0])
    return True

def bashcomplete(cli, prog_name, complete_var, complete_instr):
    if False:
        for i in range(10):
            print('nop')
    if '_' in complete_instr:
        (command, shell) = complete_instr.split('_', 1)
    else:
        command = complete_instr
        shell = 'bash'
    if command == 'source':
        echo(get_completion_script(prog_name, complete_var, shell))
        return True
    elif command == 'complete':
        if shell == 'fish':
            return do_complete_fish(cli, prog_name)
        elif shell in {'bash', 'zsh'}:
            return do_complete(cli, prog_name, shell == 'zsh')
    return False