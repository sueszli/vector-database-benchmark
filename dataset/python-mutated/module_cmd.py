"""
This module contains routines related to the module command for accessing and
parsing environment modules.
"""
import os
import re
import subprocess
import llnl.util.tty as tty
module_change_commands = ['load', 'swap', 'unload', 'purge', 'use', 'unuse']
awk_cmd = 'awk \'BEGIN{for(name in ENVIRON)printf("%s=%s%c", name, ENVIRON[name], 0)}\''

def module(*args, **kwargs):
    if False:
        print('Hello World!')
    module_cmd = kwargs.get('module_template', 'module ' + ' '.join(args))
    if args[0] in module_change_commands:
        module_cmd += ' >/dev/null 2>&1; ' + awk_cmd
        module_p = subprocess.Popen(module_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
        environ = {}
        output = module_p.communicate()[0]
        for entry in output.strip(b'\x00').split(b'\x00'):
            parts = entry.split(b'=', 1)
            if len(parts) != 2:
                continue
            environ[parts[0]] = parts[1]
        os.environ.clear()
        os.environb.update(environ)
    else:
        module_p = subprocess.Popen(module_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
        return str(module_p.communicate()[0].decode())

def load_module(mod):
    if False:
        while True:
            i = 10
    'Takes a module name and removes modules until it is possible to\n    load that module. It then loads the provided module. Depends on the\n    modulecmd implementation of modules used in cray and lmod.\n    '
    tty.debug('module_cmd.load_module: {0}'.format(mod))
    text = module('show', mod).split()
    for (i, word) in enumerate(text):
        if word == 'conflict':
            module('unload', text[i + 1])
    module('load', mod)

def get_path_args_from_module_line(line):
    if False:
        print('Hello World!')
    if '(' in line and ')' in line:
        comma_index = line.index(',')
        cline = line[comma_index:]
        try:
            quote_index = min((cline.find(q) for q in ['"', "'"] if q in cline))
            lua_quote = cline[quote_index]
        except ValueError:
            raise ValueError('No lua quote symbol found in lmod module line.')
        words_and_symbols = line.split(lua_quote)
        path_arg = words_and_symbols[-2]
    else:
        words = line.split()
        if len(words) > 2:
            path_arg = words[2]
        else:
            return []
    paths = path_arg.split(':')
    return paths

def path_from_modules(modules):
    if False:
        for i in range(10):
            print('nop')
    'Inspect a list of Tcl modules for entries that indicate the absolute\n    path at which the library supported by said module can be found.\n\n    Args:\n        modules (list): module files to be loaded to get an external package\n\n    Returns:\n        Guess of the prefix path where the package\n    '
    assert isinstance(modules, list), 'the "modules" argument must be a list'
    best_choice = None
    for module_name in modules:
        text = module('show', module_name).split('\n')
        candidate_path = get_path_from_module_contents(text, module_name)
        if candidate_path and (not os.path.exists(candidate_path)):
            msg = 'Extracted path from module does not exist [module={0}, path={1}]'
            tty.warn(msg.format(module_name, candidate_path))
        best_choice = candidate_path or best_choice
    return best_choice

def get_path_from_module_contents(text, module_name):
    if False:
        for i in range(10):
            print('nop')
    tty.debug('Module name: ' + module_name)
    pkg_var_prefix = module_name.replace('-', '_').upper()
    components = pkg_var_prefix.split('/')
    if len(components) > 1:
        pkg_var_prefix = components[-2]
    tty.debug('Package directory variable prefix: ' + pkg_var_prefix)
    path_occurrences = {}

    def strip_path(path, endings):
        if False:
            return 10
        for ending in endings:
            if path.endswith(ending):
                return path[:-len(ending)]
            if path.endswith(ending + '/'):
                return path[:-(len(ending) + 1)]
        return path

    def match_pattern_and_strip(line, pattern, strip=[]):
        if False:
            print('Hello World!')
        if re.search(pattern, line):
            paths = get_path_args_from_module_line(line)
            for path in paths:
                path = strip_path(path, strip)
                path_occurrences[path] = path_occurrences.get(path, 0) + 1

    def match_flag_and_strip(line, flag, strip=[]):
        if False:
            print('Hello World!')
        flag_idx = line.find(flag)
        if flag_idx >= 0:
            separators = (' ', '"', "'")
            occurrences = [line.find(s, flag_idx) for s in separators]
            indices = [idx for idx in occurrences if idx >= 0]
            if indices:
                path = line[flag_idx + len(flag):min(indices)]
            else:
                path = line[flag_idx + len(flag):]
            path = strip_path(path, strip)
            path_occurrences[path] = path_occurrences.get(path, 0) + 1
    lib_endings = ['/lib64', '/lib']
    bin_endings = ['/bin']
    man_endings = ['/share/man', '/man']
    for line in text:
        pattern = '\\W(CRAY_)?LD_LIBRARY_PATH'
        match_pattern_and_strip(line, pattern, lib_endings)
        pattern = '\\W{0}_DIR'.format(pkg_var_prefix)
        match_pattern_and_strip(line, pattern)
        pattern = '\\W{0}_ROOT'.format(pkg_var_prefix)
        match_pattern_and_strip(line, pattern)
        pattern = '\\WPATH'
        match_pattern_and_strip(line, pattern, bin_endings)
        pattern = 'MANPATH'
        match_pattern_and_strip(line, pattern, man_endings)
        match_flag_and_strip(line, '-rpath', lib_endings)
        match_flag_and_strip(line, '-L', lib_endings)
    if len(path_occurrences) > 0:
        return max(path_occurrences.items(), key=lambda x: x[1])[0]
    return None