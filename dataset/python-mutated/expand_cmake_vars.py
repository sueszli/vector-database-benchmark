"""Expands CMake variables in a text file."""
import re
import sys
_CMAKE_DEFINE_REGEX = re.compile('\\s*#cmakedefine\\s+([A-Za-z_0-9]*)(\\s.*)?$')
_CMAKE_DEFINE01_REGEX = re.compile('\\s*#cmakedefine01\\s+([A-Za-z_0-9]*)')
_CMAKE_VAR_REGEX = re.compile('\\${([A-Za-z_0-9]*)}')
_CMAKE_ATVAR_REGEX = re.compile('@([A-Za-z_0-9]*)@')

def _parse_args(argv):
    if False:
        i = 10
        return i + 15
    'Parses arguments with the form KEY=VALUE into a dictionary.'
    result = {}
    for arg in argv:
        (k, v) = arg.split('=')
        result[k] = v
    return result

def _expand_variables(input_str, cmake_vars):
    if False:
        for i in range(10):
            print('nop')
    "Expands ${VARIABLE}s and @VARIABLE@s in 'input_str', using dictionary 'cmake_vars'.\n\n  Args:\n    input_str: the string containing ${VARIABLE} or @VARIABLE@ expressions to expand.\n    cmake_vars: a dictionary mapping variable names to their values.\n\n  Returns:\n    The expanded string.\n  "

    def replace(match):
        if False:
            print('Hello World!')
        if match.group(1) in cmake_vars:
            return cmake_vars[match.group(1)]
        return ''
    return _CMAKE_ATVAR_REGEX.sub(replace, _CMAKE_VAR_REGEX.sub(replace, input_str))

def _expand_cmakedefines(line, cmake_vars):
    if False:
        return 10
    "Expands #cmakedefine declarations, using a dictionary 'cmake_vars'."
    match = _CMAKE_DEFINE_REGEX.match(line)
    if match:
        name = match.group(1)
        suffix = match.group(2) or ''
        if name in cmake_vars:
            return '#define {}{}\n'.format(name, _expand_variables(suffix, cmake_vars))
        else:
            return '/* #undef {} */\n'.format(name)
    match = _CMAKE_DEFINE01_REGEX.match(line)
    if match:
        name = match.group(1)
        value = cmake_vars.get(name, '0')
        return '#define {} {}\n'.format(name, value)
    return _expand_variables(line, cmake_vars)

def main():
    if False:
        print('Hello World!')
    cmake_vars = _parse_args(sys.argv[1:])
    for line in sys.stdin:
        sys.stdout.write(_expand_cmakedefines(line, cmake_vars))
if __name__ == '__main__':
    main()