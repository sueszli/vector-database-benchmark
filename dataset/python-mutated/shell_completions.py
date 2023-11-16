"""jc - JSON Convert shell_completions module"""
from string import Template
from .cli_data import long_options_map
from .lib import all_parser_info
bash_template = Template('_jc()\n{\n    local cur prev words cword jc_commands jc_parsers jc_options \\\n          jc_about_options jc_about_mod_options jc_help_options jc_special_options\n\n    jc_commands=(${bash_commands})\n    jc_parsers=(${bash_parsers})\n    jc_options=(${bash_options})\n    jc_about_options=(${bash_about_options})\n    jc_about_mod_options=(${bash_about_mod_options})\n    jc_help_options=(${bash_help_options})\n    jc_special_options=(${bash_special_options})\n\n    COMPREPLY=()\n    _get_comp_words_by_ref cur prev words cword\n\n    # if jc_about_options are found anywhere in the line, then only complete from jc_about_mod_options\n    for i in "$${words[@]::$${#words[@]}-1}"; do\n        if [[ " $${jc_about_options[*]} " =~ " $${i} " ]]; then\n            COMPREPLY=( $$( compgen -W "$${jc_about_mod_options[*]}" \\\n            -- "$${cur}" ) )\n            return 0\n        fi\n    done\n\n    # if jc_help_options and a parser are found anywhere in the line, then no more completions\n    if\n        (\n            for i in "$${words[@]::$${#words[@]}-1}"; do\n                if [[ " $${jc_help_options[*]} " =~ " $${i} " ]]; then\n                    return 0\n                fi\n            done\n            return 1\n        ) && (\n            for i in "$${words[@]::$${#words[@]}-1}"; do\n                if [[ " $${jc_parsers[*]} " =~ " $${i} " ]]; then\n                    return 0\n                fi\n            done\n            return 1\n        ); then\n        return 0\n    fi\n\n    # if jc_help_options are found anywhere in the line, then only complete with parsers\n    for i in "$${words[@]::$${#words[@]}-1}"; do\n        if [[ " $${jc_help_options[*]} " =~ " $${i} " ]]; then\n            COMPREPLY=( $$( compgen -W "$${jc_parsers[*]}" \\\n            -- "$${cur}" ) )\n            return 0\n        fi\n    done\n\n    # if special options are found anywhere in the line, then no more completions\n    for i in "$${words[@]::$${#words[@]}-1}"; do\n        if [[ " $${jc_special_options[*]} " =~ " $${i} " ]]; then\n            return 0\n        fi\n    done\n\n    # if magic command is found anywhere in the line, use called command\'s autocompletion\n    for i in "$${words[@]::$${#words[@]}-1}"; do\n        if [[ " $${jc_commands[*]} " =~ " $${i} " ]]; then\n            _command\n            return 0\n        fi\n    done\n\n    # if "/pr[oc]" (magic for Procfile parsers) is in the current word, complete with files/directories in the path\n    if [[ "$${cur}" =~ "/pr" ]]; then\n        _filedir\n        return 0\n    fi\n\n    # if a parser arg is found anywhere in the line, only show options and help options\n    for i in "$${words[@]::$${#words[@]}-1}"; do\n        if [[ " $${jc_parsers[*]} " =~ " $${i} " ]]; then\n            COMPREPLY=( $$( compgen -W "$${jc_options[*]} $${jc_help_options[*]}" \\\n            -- "$${cur}" ) )\n            return 0\n        fi\n    done\n\n    # default completion\n    COMPREPLY=( $$( compgen -W "$${jc_options[*]} $${jc_about_options[*]} $${jc_help_options[*]} $${jc_special_options[*]} $${jc_parsers[*]} $${jc_commands[*]}" \\\n        -- "$${cur}" ) )\n} &&\ncomplete -F _jc jc\n')
zsh_template = Template('#compdef jc\n\n_jc() {\n    local -a jc_commands jc_commands_describe \\\n             jc_parsers jc_parsers_describe \\\n             jc_options jc_options_describe \\\n             jc_about_options jc_about_options_describe \\\n             jc_about_mod_options jc_about_mod_options_describe \\\n             jc_help_options jc_help_options_describe \\\n             jc_special_options jc_special_options_describe\n\n    jc_commands=(${zsh_commands})\n    jc_commands_describe=(\n        ${zsh_commands_describe}\n    )\n    jc_parsers=(${zsh_parsers})\n    jc_parsers_describe=(\n        ${zsh_parsers_describe}\n    )\n    jc_options=(${zsh_options})\n    jc_options_describe=(\n        ${zsh_options_describe}\n    )\n    jc_about_options=(${zsh_about_options})\n    jc_about_options_describe=(\n        ${zsh_about_options_describe}\n    )\n    jc_about_mod_options=(${zsh_about_mod_options})\n    jc_about_mod_options_describe=(\n        ${zsh_about_mod_options_describe}\n    )\n    jc_help_options=(${zsh_help_options})\n    jc_help_options_describe=(\n        ${zsh_help_options_describe}\n    )\n    jc_special_options=(${zsh_special_options})\n    jc_special_options_describe=(\n        ${zsh_special_options_describe}\n    )\n\n    # if jc_about_options are found anywhere in the line, then only complete from jc_about_mod_options\n    for i in $${words:0:-1}; do\n        if (( $$jc_about_options[(Ie)$${i}] )); then\n            _describe \'commands\' jc_about_mod_options_describe\n            return 0\n        fi\n    done\n\n    # if jc_help_options and a parser are found anywhere in the line, then no more completions\n     if\n        (\n            for i in $${words:0:-1}; do\n                if (( $$jc_help_options[(Ie)$${i}] )); then\n                    return 0\n                fi\n            done\n            return 1\n        ) && (\n            for i in $${words:0:-1}; do\n                if (( $$jc_parsers[(Ie)$${i}] )); then\n                    return 0\n                fi\n            done\n            return 1\n        ); then\n        return 0\n    fi\n\n    # if jc_help_options are found anywhere in the line, then only complete with parsers\n    for i in $${words:0:-1}; do\n        if (( $$jc_help_options[(Ie)$${i}] )); then\n            _describe \'commands\' jc_parsers_describe\n            return 0\n        fi\n    done\n\n    # if special options are found anywhere in the line, then no more completions\n    for i in $${words:0:-1}; do\n        if (( $$jc_special_options[(Ie)$${i}] )); then\n            return 0\n        fi\n    done\n\n    # if magic command is found anywhere in the line, use called command\'s autocompletion\n    for i in $${words:0:-1}; do\n        if (( $$jc_commands[(Ie)$${i}] )); then\n            # hack to remove options between jc and the magic command\n            shift $$(( $${#words} - 2 )) words\n            words[1,0]=(jc)\n            CURRENT=$${#words}\n\n            # run the magic command\'s completions\n            _arguments \'*::arguments:_normal\'\n            return 0\n        fi\n    done\n\n    # if "/pr[oc]" (magic for Procfile parsers) is in the current word, complete with files/directories in the path\n    if [[ "$${words[-1]}" =~ "/pr" ]]; then\n        # run files completion\n        _files\n        return 0\n    fi\n\n    # if a parser arg is found anywhere in the line, only show options and help options\n    for i in $${words:0:-1}; do\n        if (( $$jc_parsers[(Ie)$${i}] )); then\n            _describe \'commands\' jc_options_describe -- jc_help_options_describe\n            return 0\n        fi\n    done\n\n    # default completion\n    _describe \'commands\' jc_options_describe -- jc_about_options_describe -- jc_help_options_describe -- jc_special_options_describe -- jc_parsers_describe -- jc_commands_describe\n}\n\n_jc\n')
about_options = ['--about', '-a']
about_mod_options = ['--pretty', '-p', '--yaml-out', '-y', '--monochrome', '-m', '--force-color', '-C']
help_options = ['--help', '-h']
special_options = ['--version', '-v', '--bash-comp', '-B', '--zsh-comp', '-Z']

def get_commands():
    if False:
        print('Hello World!')
    command_list = []
    for cmd in all_parser_info():
        if 'magic_commands' in cmd:
            command_list.extend(cmd['magic_commands'])
    return sorted(list(set([i.split()[0] for i in command_list])))

def get_options():
    if False:
        while True:
            i = 10
    options_list = []
    for opt in long_options_map:
        options_list.append(opt)
        options_list.append('-' + long_options_map[opt][0])
    return options_list

def get_parsers():
    if False:
        print('Hello World!')
    p_list = []
    for cmd in all_parser_info(show_hidden=True):
        if 'argument' in cmd:
            p_list.append(cmd['argument'])
    return p_list

def get_parsers_descriptions():
    if False:
        print('Hello World!')
    pd_list = []
    for p in all_parser_info(show_hidden=True):
        if 'description' in p:
            pd_list.append(f"'{p['argument']}:{p['description']}'")
    return pd_list

def get_zsh_command_descriptions(command_list):
    if False:
        i = 10
        return i + 15
    zsh_commands = []
    for cmd in command_list:
        zsh_commands.append(f"""'{cmd}:run "{cmd}" command with magic syntax.'""")
    return zsh_commands

def get_descriptions(opt_list):
    if False:
        while True:
            i = 10
    'Return a list of options:description items.'
    opt_desc_list = []
    for item in opt_list:
        if item in long_options_map:
            opt_desc_list.append(f"'{item}:{long_options_map[item][1]}'")
            continue
        for (k, v) in long_options_map.items():
            if item[1:] == v[0]:
                opt_desc_list.append(f"'{item}:{v[1]}'")
                continue
    return opt_desc_list

def bash_completion():
    if False:
        for i in range(10):
            print('nop')
    parsers_str = ' '.join(get_parsers())
    opts_no_special = get_options()
    for s_option in special_options:
        opts_no_special.remove(s_option)
    for a_option in about_options:
        opts_no_special.remove(a_option)
    for h_option in help_options:
        opts_no_special.remove(h_option)
    options_str = ' '.join(opts_no_special)
    about_options_str = ' '.join(about_options)
    about_mod_options_str = ' '.join(about_mod_options)
    help_options_str = ' '.join(help_options)
    special_options_str = ' '.join(special_options)
    commands_str = ' '.join(get_commands())
    return bash_template.substitute(bash_parsers=parsers_str, bash_special_options=special_options_str, bash_about_options=about_options_str, bash_about_mod_options=about_mod_options_str, bash_help_options=help_options_str, bash_options=options_str, bash_commands=commands_str)

def zsh_completion():
    if False:
        for i in range(10):
            print('nop')
    parsers_str = ' '.join(get_parsers())
    parsers_describe = '\n        '.join(get_parsers_descriptions())
    opts_no_special = get_options()
    for s_option in special_options:
        opts_no_special.remove(s_option)
    for a_option in about_options:
        opts_no_special.remove(a_option)
    for h_option in help_options:
        opts_no_special.remove(h_option)
    options_str = ' '.join(opts_no_special)
    options_describe = '\n        '.join(get_descriptions(opts_no_special))
    about_options_str = ' '.join(about_options)
    about_options_describe = '\n        '.join(get_descriptions(about_options))
    about_mod_options_str = ' '.join(about_mod_options)
    about_mod_options_describe = '\n        '.join(get_descriptions(about_mod_options))
    help_options_str = ' '.join(help_options)
    help_options_describe = '\n        '.join(get_descriptions(help_options))
    special_options_str = ' '.join(special_options)
    special_options_describe = '\n        '.join(get_descriptions(special_options))
    commands_str = ' '.join(get_commands())
    commands_describe = '\n        '.join(get_zsh_command_descriptions(get_commands()))
    return zsh_template.substitute(zsh_parsers=parsers_str, zsh_parsers_describe=parsers_describe, zsh_special_options=special_options_str, zsh_special_options_describe=special_options_describe, zsh_about_options=about_options_str, zsh_about_options_describe=about_options_describe, zsh_about_mod_options=about_mod_options_str, zsh_about_mod_options_describe=about_mod_options_describe, zsh_help_options=help_options_str, zsh_help_options_describe=help_options_describe, zsh_options=options_str, zsh_options_describe=options_describe, zsh_commands=commands_str, zsh_commands_describe=commands_describe)