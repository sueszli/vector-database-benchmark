"""This plugin generates tab completions for Beets commands for the Fish shell
<https://fishshell.com/>, including completions for Beets commands, plugin
commands, and option flags. Also generated are completions for all the album
and track fields, suggesting for example `genre:` or `album:` when querying the
Beets database. Completions for the *values* of those fields are not generated
by default but can be added via the `-e` / `--extravalues` flag. For example:
`beet fish -e genre -e albumartist`
"""
import os
from operator import attrgetter
from beets import library, ui
from beets.plugins import BeetsPlugin
from beets.ui import commands
BL_NEED2 = "complete -c beet -n '__fish_beet_needs_command' {} {}\n"
BL_USE3 = "complete -c beet -n '__fish_beet_using_command {}' {} {}\n"
BL_SUBS = 'complete -c beet -n \'__fish_at_level {} ""\' {}  {}\n'
BL_EXTRA3 = "complete -c beet -n '__fish_beet_use_extra {}' {} {}\n"
HEAD = '\nfunction __fish_beet_needs_command\n    set cmd (commandline -opc)\n    if test (count $cmd) -eq 1\n        return 0\n    end\n    return 1\nend\n\nfunction __fish_beet_using_command\n    set cmd (commandline -opc)\n    set needle (count $cmd)\n    if test $needle -gt 1\n        if begin test $argv[1] = $cmd[2];\n            and not contains -- $cmd[$needle] $FIELDS; end\n                return 0\n        end\n    end\n    return 1\nend\n\nfunction __fish_beet_use_extra\n    set cmd (commandline -opc)\n    set needle (count $cmd)\n    if test $argv[2]  = $cmd[$needle]\n        return 0\n    end\n    return 1\nend\n'

class FishPlugin(BeetsPlugin):

    def commands(self):
        if False:
            return 10
        cmd = ui.Subcommand('fish', help='generate Fish shell tab completions')
        cmd.func = self.run
        cmd.parser.add_option('-f', '--noFields', action='store_true', default=False, help='omit album/track field completions')
        cmd.parser.add_option('-e', '--extravalues', action='append', type='choice', choices=library.Item.all_keys() + library.Album.all_keys(), help='include specified field *values* in completions')
        cmd.parser.add_option('-o', '--output', default='~/.config/fish/completions/beet.fish', help='where to save the script. default: ~/.config/fish/completions')
        return [cmd]

    def run(self, lib, opts, args):
        if False:
            while True:
                i = 10
        completion_file_path = os.path.expanduser(opts.output)
        completion_dir = os.path.dirname(completion_file_path)
        if completion_dir != '':
            os.makedirs(completion_dir, exist_ok=True)
        nobasicfields = opts.noFields
        extravalues = opts.extravalues
        beetcmds = sorted(commands.default_commands + commands.plugins.commands(), key=attrgetter('name'))
        fields = sorted(set(library.Album.all_keys() + library.Item.all_keys()))
        cmd_names_help = []
        for cmd in beetcmds:
            names = list(cmd.aliases)
            names.append(cmd.name)
            for name in names:
                cmd_names_help.append((name, cmd.help))
        totstring = HEAD + '\n'
        totstring += get_cmds_list([name[0] for name in cmd_names_help])
        totstring += '' if nobasicfields else get_standard_fields(fields)
        totstring += get_extravalues(lib, extravalues) if extravalues else ''
        totstring += '\n' + '# ====== {} ====='.format('setup basic beet completion') + '\n' * 2
        totstring += get_basic_beet_options()
        totstring += '\n' + '# ====== {} ====='.format('setup field completion for subcommands') + '\n'
        totstring += get_subcommands(cmd_names_help, nobasicfields, extravalues)
        totstring += get_all_commands(beetcmds)
        with open(completion_file_path, 'w') as fish_file:
            fish_file.write(totstring)

def _escape(name):
    if False:
        print('Hello World!')
    if name == '?':
        name = '\\' + name
    return name

def get_cmds_list(cmds_names):
    if False:
        for i in range(10):
            print('nop')
    substr = ''
    substr += 'set CMDS ' + ' '.join(cmds_names) + '\n' * 2
    return substr

def get_standard_fields(fields):
    if False:
        return 10
    fields = (field + ':' for field in fields)
    substr = ''
    substr += 'set FIELDS ' + ' '.join(fields) + '\n' * 2
    return substr

def get_extravalues(lib, extravalues):
    if False:
        i = 10
        return i + 15
    word = ''
    values_set = get_set_of_values_for_field(lib, extravalues)
    for fld in extravalues:
        extraname = fld.upper() + 'S'
        word += 'set  ' + extraname + ' ' + ' '.join(sorted(values_set[fld])) + '\n' * 2
    return word

def get_set_of_values_for_field(lib, fields):
    if False:
        for i in range(10):
            print('nop')
    fields_dict = {}
    for each in fields:
        fields_dict[each] = set()
    for item in lib.items():
        for field in fields:
            fields_dict[field].add(wrap(item[field]))
    return fields_dict

def get_basic_beet_options():
    if False:
        i = 10
        return i + 15
    word = BL_NEED2.format('-l format-item', "-f -d 'print with custom format'") + BL_NEED2.format('-l format-album', "-f -d 'print with custom format'") + BL_NEED2.format('-s  l  -l library', "-f -r -d 'library database file to use'") + BL_NEED2.format('-s  d  -l directory', "-f -r -d 'destination music directory'") + BL_NEED2.format('-s  v  -l verbose', "-f -d 'print debugging information'") + BL_NEED2.format('-s  c  -l config', "-f -r -d 'path to configuration file'") + BL_NEED2.format('-s  h  -l help', "-f -d 'print this help message and exit'")
    return word

def get_subcommands(cmd_name_and_help, nobasicfields, extravalues):
    if False:
        for i in range(10):
            print('nop')
    word = ''
    for (cmdname, cmdhelp) in cmd_name_and_help:
        cmdname = _escape(cmdname)
        word += '\n' + '# ------ {} -------'.format('fieldsetups for  ' + cmdname) + '\n'
        word += BL_NEED2.format('-a ' + cmdname, '-f ' + '-d ' + wrap(clean_whitespace(cmdhelp)))
        if nobasicfields is False:
            word += BL_USE3.format(cmdname, '-a ' + wrap('$FIELDS'), '-f ' + '-d ' + wrap('fieldname'))
        if extravalues:
            for f in extravalues:
                setvar = wrap('$' + f.upper() + 'S')
                word += ' '.join(BL_EXTRA3.format(cmdname + ' ' + f + ':', '-f ' + '-A ' + '-a ' + setvar, '-d ' + wrap(f)).split()) + '\n'
    return word

def get_all_commands(beetcmds):
    if False:
        print('Hello World!')
    word = ''
    for cmd in beetcmds:
        names = list(cmd.aliases)
        names.append(cmd.name)
        for name in names:
            name = _escape(name)
            word += '\n'
            word += '\n' * 2 + '# ====== {} ====='.format('completions for  ' + name) + '\n'
            for option in cmd.parser._get_all_options()[1:]:
                cmd_l = ' -l ' + option._long_opts[0].replace('--', '') if option._long_opts else ''
                cmd_s = ' -s ' + option._short_opts[0].replace('-', '') if option._short_opts else ''
                cmd_need_arg = ' -r ' if option.nargs in [1] else ''
                cmd_helpstr = ' -d ' + wrap(' '.join(option.help.split())) if option.help else ''
                cmd_arglist = ' -a ' + wrap(' '.join(option.choices)) if option.choices else ''
                word += ' '.join(BL_USE3.format(name, cmd_need_arg + cmd_s + cmd_l + ' -f ' + cmd_arglist, cmd_helpstr).split()) + '\n'
            word = word + ' '.join(BL_USE3.format(name, '-s ' + 'h ' + '-l ' + 'help' + ' -f ', '-d ' + wrap('print help') + '\n').split())
    return word

def clean_whitespace(word):
    if False:
        return 10
    return ' '.join(word.split())

def wrap(word):
    if False:
        while True:
            i = 10
    sptoken = '"'
    if '"' in word and "'" in word:
        word.replace('"', sptoken)
        return '"' + word + '"'
    tok = '"' if "'" in word else "'"
    return tok + word + tok