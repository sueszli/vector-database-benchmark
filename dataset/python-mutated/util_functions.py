""" Utility functions for gr_modtool """
import re
import sys
try:
    import readline
    have_readline = True
except ImportError:
    have_readline = False

def append_re_line_sequence(filename, linepattern, newline):
    if False:
        for i in range(10):
            print('nop')
    " Detects the re 'linepattern' in the file. After its last occurrence,\n    paste 'newline'. If the pattern does not exist, append the new line\n    to the file. Then, write. "
    with open(filename, 'r') as f:
        oldfile = f.read()
    lines = re.findall(linepattern, oldfile, flags=re.MULTILINE)
    if len(lines) == 0:
        with open(filename, 'a') as f:
            f.write(newline)
        return
    last_line = lines[-1]
    newfile = oldfile.replace(last_line, last_line + newline + '\n')
    with open(filename, 'w') as f:
        f.write(newfile)

def remove_pattern_from_file(filename, pattern):
    if False:
        return 10
    ' Remove all occurrences of a given pattern from a file. '
    with open(filename, 'r') as f:
        oldfile = f.read()
    pattern = re.compile(pattern, re.MULTILINE)
    with open(filename, 'w') as f:
        f.write(pattern.sub('', oldfile))

def str_to_fancyc_comment(text):
    if False:
        return 10
    ' Return a string as a C formatted comment. '
    l_lines = text.splitlines()
    if len(l_lines[0]) == 0:
        outstr = '/*\n'
    else:
        outstr = '/* ' + l_lines[0] + '\n'
    for line in l_lines[1:]:
        if len(line) == 0:
            outstr += ' *\n'
        else:
            outstr += ' * ' + line + '\n'
    outstr += ' */\n'
    return outstr

def str_to_python_comment(text):
    if False:
        print('Hello World!')
    ' Return a string as a Python formatted comment. '
    l_lines = text.splitlines()
    if len(l_lines[0]) == 0:
        outstr = '#\n'
    else:
        outstr = '# ' + l_lines[0] + '\n'
    for line in l_lines[1:]:
        if len(line) == 0:
            outstr += '#\n'
        else:
            outstr += '# ' + line + '\n'
    outstr += '#\n'
    return outstr

def strip_default_values(string):
    if False:
        print('Hello World!')
    ' Strip default values from a C++ argument list. '
    return re.sub(' *=[^,)]*', '', string)

def strip_arg_types(string):
    if False:
        return 10
    '"\n    Strip the argument types from a list of arguments.\n    Example: "int arg1, double arg2" -> "arg1, arg2"\n    Note that some types have qualifiers, which also are part of\n    the type, e.g. "const std::string &name" -> "name", or\n    "const char *str" -> "str".\n    '
    string = strip_default_values(string)
    return ', '.join([part.strip().split(' ')[-1] for part in string.split(',')]).replace('*', '').replace('&', '')

def strip_arg_types_grc(string):
    if False:
        print('Hello World!')
    '" Strip the argument types from a list of arguments for GRC make tag.\n    Example: "int arg1, double arg2" -> "$arg1, $arg2" '
    if len(string) == 0:
        return ''
    else:
        string = strip_default_values(string)
        return ', '.join(['${' + part.strip().split(' ')[-1] + '}' for part in string.split(',')])

def get_modname():
    if False:
        print('Hello World!')
    " Grep the current module's name from gnuradio.project or CMakeLists.txt "
    modname_trans = {'howto-write-a-block': 'howto'}
    try:
        with open('gnuradio.project', 'r') as f:
            prfile = f.read()
        regexp = 'projectname\\s*=\\s*([a-zA-Z0-9-_]+)$'
        return re.search(regexp, prfile, flags=re.MULTILINE).group(1).strip()
    except IOError:
        pass
    with open('CMakeLists.txt', 'r') as f:
        cmfile = f.read()
    regexp = '(project\\s*\\(\\s*|GR_REGISTER_COMPONENT\\(")gr-(?P<modname>[a-zA-Z0-9-_]+)(\\s*(CXX)?|" ENABLE)'
    try:
        modname = re.search(regexp, cmfile, flags=re.MULTILINE).group('modname').strip()
        if modname in list(modname_trans.keys()):
            modname = modname_trans[modname]
        return modname
    except AttributeError:
        return None

def get_block_names(pattern, modname):
    if False:
        i = 10
        return i + 15
    ' Return a list of block names belonging to modname that matches the regex pattern. '
    blocknames = []
    reg = re.compile(pattern)
    fname_re = re.compile('[a-zA-Z]\\w+\\.\\w{1,5}$')
    with open(f'include/gnuradio/{modname}/CMakeLists.txt', 'r') as f:
        for line in f.read().splitlines():
            if len(line.strip()) == 0 or line.strip()[0] == '#':
                continue
            for word in re.split('[ /)(\t\n\r\x0c\x0b]', line):
                if fname_re.match(word) and reg.search(word):
                    blocknames.append(word.strip('.h'))
    return blocknames

def is_number(s):
    if False:
        for i in range(10):
            print('nop')
    ' Return True if the string s contains a number. '
    try:
        float(s)
        return True
    except ValueError:
        return False

def ask_yes_no(question, default):
    if False:
        for i in range(10):
            print('nop')
    ' Asks a binary question. Returns True for yes, False for no.\n    default is given as a boolean. '
    question += {True: ' [Y/n] ', False: ' [y/N] '}[default]
    if input(question).lower() != {True: 'n', False: 'y'}[default]:
        return default
    else:
        return not default

class SequenceCompleter(object):
    """ A simple completer function wrapper to be used with readline, e.g.
    option_iterable = ("search", "seek", "destroy")
    readline.set_completer(SequenceCompleter(option_iterable).completefunc)

    Typical usage is with the `with` statement. Restores the previous completer
    at exit, thus nestable.
    """

    def __init__(self, sequence=None):
        if False:
            i = 10
            return i + 15
        self._seq = sequence or []
        self._tmp_matches = []

    def completefunc(self, text, state):
        if False:
            i = 10
            return i + 15
        if not text and state < len(self._seq):
            return self._seq[state]
        if not state:
            self._tmp_matches = [candidate for candidate in self._seq if candidate.startswith(text)]
        if state < len(self._tmp_matches):
            return self._tmp_matches[state]

    def __enter__(self):
        if False:
            print('Hello World!')
        if have_readline:
            self._old_completer = readline.get_completer()
            readline.set_completer(self.completefunc)
            readline.parse_and_bind('tab: complete')

    def __exit__(self, exception_type, exception_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        if have_readline:
            readline.set_completer(self._old_completer)