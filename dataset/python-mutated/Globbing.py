import os
import platform
import re
from functools import lru_cache
from coala_utils.decorators import yield_once
from coalib.misc.Constants import GLOBBING_SPECIAL_CHARS

def _end_of_set_index(string, start_index):
    if False:
        i = 10
        return i + 15
    '\n    Returns the position of the appropriate closing bracket for a glob set in\n    string.\n\n    :param string:      Glob string with wildcards\n    :param start_index: Index at which the set starts, meaning the position\n                        right behind the opening bracket\n    :return:            Position of appropriate closing bracket\n    '
    length = len(string)
    closing_index = start_index
    if closing_index < length and string[closing_index] == '!':
        closing_index += 1
    if closing_index < length:
        closing_index += 1
    while closing_index < length and string[closing_index] != ']':
        closing_index += 1
    return closing_index

def glob_escape(input_string):
    if False:
        print('Hello World!')
    "\n    Escapes the given string with ``[c]`` pattern. Examples:\n\n    >>> from coalib.parsing.Globbing import glob_escape\n    >>> glob_escape('test (1)')\n    'test [(]1[)]'\n    >>> glob_escape('test folder?')\n    'test folder[?]'\n    >>> glob_escape('test*folder')\n    'test[*]folder'\n\n    :param input_string: String that is to be escaped with ``[ ]``.\n    :return:             Escaped string in which all the special glob characters\n                         ``()[]|?*`` are escaped.\n    "
    return re.sub('(?P<char>[' + re.escape(GLOBBING_SPECIAL_CHARS) + '])', '[\\g<char>]', input_string)

def _position_is_bracketed(string, position):
    if False:
        return 10
    '\n    Tests whether the char at string[position] is inside a valid pair of\n    brackets (and therefore loses its special meaning)\n\n    :param string:   Glob string with wildcards\n    :param position: Position of a char in string\n    :return:         Whether or not the char is inside a valid set of brackets\n    '
    position = len(string[:position])
    (index, length) = (0, len(string))
    while index < position:
        char = string[index]
        index += 1
        if char == '[':
            closing_index = _end_of_set_index(string, index)
            if closing_index < length:
                if index <= position < closing_index:
                    return True
                index = closing_index + 1
            else:
                return False
    return False

def _boundary_of_alternatives_indices(pattern):
    if False:
        i = 10
        return i + 15
    '\n    Determines the location of a set of alternatives in a glob pattern.\n    Alternatives are defined by a matching set of non-bracketed parentheses.\n\n    :param pattern: Glob pattern with wildcards.\n    :return:        Indices of the innermost set of matching non-bracketed\n                    parentheses in a tuple. The Index of a missing parenthesis\n                    will be passed as None.\n    '
    end_pos = None
    for match in re.finditer('\\)', pattern):
        if not _position_is_bracketed(pattern, match.start()):
            end_pos = match.start()
            break
    start_pos = None
    for match in re.finditer('\\(', pattern[:end_pos]):
        if not _position_is_bracketed(pattern, match.start()):
            start_pos = match.end()
    return (start_pos, end_pos)

@yield_once
def _iter_choices(pattern):
    if False:
        while True:
            i = 10
    "\n    Iterate through each choice of an alternative. Splits pattern on '|'s if\n    they are not bracketed.\n\n    :param pattern: String of choices separated by '|'s\n    :return:        Iterator that yields parts of string separated by\n                    non-bracketed '|'s\n    "
    start_pos = 0
    split_pos_list = [match.start() for match in re.finditer('\\|', pattern)]
    split_pos_list.append(len(pattern))
    for end_pos in split_pos_list:
        if not _position_is_bracketed(pattern, end_pos):
            yield pattern[start_pos:end_pos]
            start_pos = end_pos + 1

@yield_once
def _iter_alternatives(pattern):
    if False:
        while True:
            i = 10
    '\n    Iterates through all glob patterns that can be obtained by combination of\n    all choices for each alternative\n\n    :param pattern: Glob pattern with wildcards\n    :return:        Iterator that yields all glob patterns without alternatives\n                    that can be created from the given pattern containing them.\n    '
    (start_pos, end_pos) = _boundary_of_alternatives_indices(pattern)
    if None in (start_pos, end_pos):
        yield pattern
    else:
        for choice in _iter_choices(pattern[start_pos:end_pos]):
            variant = pattern[:start_pos - 1] + choice + pattern[end_pos + 1:]
            for glob_pattern in _iter_alternatives(variant):
                yield glob_pattern

def translate(pattern):
    if False:
        while True:
            i = 10
    '\n    Translates a pattern into a regular expression.\n\n    :param pattern: Glob pattern with wildcards\n    :return:        Regular expression with the same meaning\n    '
    (index, length) = (0, len(pattern))
    regex = ''
    while index < length:
        char = pattern[index]
        index += 1
        if char == '*':
            if index < length and pattern[index] == '*':
                regex += '.*'
            elif platform.system() == 'Windows':
                regex += '[^/\\\\]*'
            else:
                regex += '[^' + re.escape(os.sep) + ']*'
        elif char == '?':
            regex += '.'
        elif char == '[':
            closing_index = _end_of_set_index(pattern, index)
            if closing_index >= length:
                regex += '\\['
            else:
                sequence = pattern[index:closing_index].replace('\\', '\\\\')
                index = closing_index + 1
                if sequence[0] == '!':
                    sequence = '^' + sequence[1:]
                elif sequence[0] == '^':
                    sequence = '\\' + sequence
                regex += '[' + sequence + ']'
        else:
            regex = regex + re.escape(char)
    return '(?ms)' + regex + '\\Z'

def fnmatch(name, globs):
    if False:
        return 10
    "\n    Tests whether name matches one of the given globs.\n\n    An empty ``globs`` list always returns true.\n\n    An empty glob in ``globs`` list will match nothing and is ignored.\n\n    :param name:  File or directory name\n    :param globs: Glob string with wildcards or list of globs\n    :return:      Boolean: Whether or not name is matched by glob\n\n    Glob Syntax:\n\n    -  '[seq]':         Matches any character in seq. Cannot be empty. Any\n                        special character looses its special meaning in a set.\n    -  '[!seq]':        Matches any character not in seq. Cannot be empty. Any\n                        special character looses its special meaning in a set.\n    -  '(seq_a|seq_b)': Matches either sequence_a or sequence_b as a whole.\n                        More than two or just one sequence can be given.\n    -  '?':             Matches any single character.\n    -  '*':             Matches everything but os.sep.\n    -  '**':            Matches everything.\n    "
    globs = (globs,) if isinstance(globs, str) else tuple(globs)
    if len(globs) == 0:
        return True
    name = os.path.normcase(name)
    return any((compiled_pattern.match(name) for glob in globs for compiled_pattern in _compile_pattern(glob)))

@lru_cache()
def _compile_pattern(pattern):
    if False:
        return 10
    return tuple((re.compile(translate(os.path.normcase(os.path.expanduser(pat)))) for pat in _iter_alternatives(pattern)))

def _absolute_flat_glob(pattern):
    if False:
        while True:
            i = 10
    '\n    Glob function for a pattern that do not contain wildcards.\n\n    :pattern: File or directory path\n    :return:  Iterator that yields at most one valid file or dir name\n    '
    (dirname, basename) = os.path.split(pattern)
    if basename:
        if os.path.exists(pattern):
            yield pattern
    elif os.path.isdir(dirname):
        yield pattern
    return

def _iter_relative_dirs(dirname):
    if False:
        while True:
            i = 10
    "\n    Recursively iterates subdirectories of all levels from dirname\n\n    :param dirname: Directory name\n    :return:        Iterator that yields files and directory from the given dir\n                    and all it's (recursive) subdirectories\n    "
    if not dirname:
        dirname = os.curdir
    try:
        files_or_dirs = os.listdir(dirname)
    except os.error:
        return
    for file_or_dir in files_or_dirs:
        yield file_or_dir
        path = os.path.join(dirname, file_or_dir)
        for sub_file_or_dir in _iter_relative_dirs(path):
            yield os.path.join(file_or_dir, sub_file_or_dir)

def relative_wildcard_glob(dirname, pattern):
    if False:
        i = 10
        return i + 15
    '\n    Non-recursive glob for one directory. Accepts wildcards.\n\n    :param dirname: Directory name\n    :param pattern: Glob pattern with wildcards\n    :return:        List of files in the dir of dirname that match the pattern\n    '
    if not dirname:
        dirname = os.curdir
    try:
        if '**' in pattern:
            names = list(_iter_relative_dirs(dirname))
        else:
            names = os.listdir(dirname)
    except OSError:
        return []
    result = []
    pattern = os.path.normcase(pattern)
    match = re.compile(translate(pattern)).match
    for name in names:
        if match(os.path.normcase(name)):
            result.append(name)
    return result

def relative_flat_glob(dirname, basename):
    if False:
        i = 10
        return i + 15
    '\n    Non-recursive glob for one directory. Does not accept wildcards.\n\n    :param dirname:  Directory name\n    :param basename: Basename of a file in dir of dirname\n    :return:         List containing Basename if the file exists\n    '
    if os.path.exists(os.path.join(dirname, basename)):
        return [basename]
    return []

def relative_recursive_glob(dirname, pattern):
    if False:
        i = 10
        return i + 15
    "\n    Recursive Glob for one directory and all its (nested) subdirectories.\n    Accepts only '**' as pattern.\n\n    :param dirname: Directory name\n    :param pattern: The recursive wildcard '**'\n    :return:        Iterator that yields all the (nested) subdirectories of the\n                    given dir\n    "
    assert pattern == '**'
    if dirname:
        yield pattern[:0]
    for relative_dir in _iter_relative_dirs(dirname):
        yield relative_dir
wildcard_check_pattern = re.compile('([*?[])')

def has_wildcard(pattern):
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks whether pattern has any wildcards.\n\n    :param pattern: Glob pattern that may contain wildcards\n    :return:        Boolean: Whether or not there are wildcards in pattern\n    '
    match = wildcard_check_pattern.search(pattern)
    return match is not None

def _iglob(pattern):
    if False:
        return 10
    (dirname, basename) = os.path.split(pattern)
    if not has_wildcard(pattern):
        for file in _absolute_flat_glob(pattern):
            yield file
        return
    if basename == '**':
        relative_glob_function = relative_recursive_glob
    elif has_wildcard(basename):
        relative_glob_function = relative_wildcard_glob
    else:
        relative_glob_function = relative_flat_glob
    if not dirname:
        for file in relative_glob_function(dirname, basename):
            yield file
        return
    if dirname != pattern and has_wildcard(dirname):
        dirs = iglob(dirname)
    else:
        dirs = [dirname]
    for dirname in dirs:
        for name in relative_glob_function(dirname, basename):
            yield os.path.join(dirname, name)

@yield_once
def iglob(pattern):
    if False:
        while True:
            i = 10
    '\n    Iterates all filesystem paths that get matched by the glob pattern.\n    Syntax is equal to that of fnmatch.\n\n    :param pattern: Glob pattern with wildcards\n    :return:        Iterator that yields all file names that match pattern\n    '
    for pat in _iter_alternatives(pattern):
        pat = os.path.expanduser(pat)
        pat = os.path.normcase(pat)
        if pat.endswith(os.sep):
            for name in _iglob(pat):
                yield name
        else:
            for name in _iglob(pat):
                yield name.rstrip(os.sep)

def glob(pattern):
    if False:
        i = 10
        return i + 15
    '\n    Iterates all filesystem paths that get matched by the glob pattern.\n    Syntax is equal to that of fnmatch.\n\n    :param pattern: Glob pattern with wildcards\n    :return:        List of all file names that match pattern\n    '
    return list(iglob(pattern))