"""
Source code text utilities
"""
import re
import os
import sys
from pylsp._utils import get_eol_chars as _get_eol_chars
EOL_CHARS = (('\r\n', 'nt'), ('\n', 'posix'), ('\r', 'mac'))
CAMEL_CASE_RE = re.compile('(?<!^)(?=[A-Z])')

def get_eol_chars(text):
    if False:
        while True:
            i = 10
    '\n    Get text end-of-line (eol) characters.\n\n    If no eol chars are found, return ones based on the operating\n    system.\n\n    Parameters\n    ----------\n    text: str\n        Text to get its eol chars from\n\n    Returns\n    -------\n    eol: str or None\n        Eol found in ``text``.\n    '
    eol_chars = _get_eol_chars(text)
    if not eol_chars:
        if os.name == 'nt':
            eol_chars = '\r\n'
        elif sys.platform.startswith('linux'):
            eol_chars = '\n'
        elif sys.platform == 'darwin':
            eol_chars = '\r'
        else:
            eol_chars = '\n'
    return eol_chars

def get_os_name_from_eol_chars(eol_chars):
    if False:
        i = 10
        return i + 15
    'Return OS name from EOL characters'
    for (chars, os_name) in EOL_CHARS:
        if eol_chars == chars:
            return os_name

def get_eol_chars_from_os_name(os_name):
    if False:
        return 10
    'Return EOL characters from OS name'
    for (eol_chars, name) in EOL_CHARS:
        if name == os_name:
            return eol_chars

def has_mixed_eol_chars(text):
    if False:
        i = 10
        return i + 15
    'Detect if text has mixed EOL characters'
    eol_chars = get_eol_chars(text)
    if eol_chars is None:
        return False
    correct_text = eol_chars.join((text + eol_chars).splitlines())
    return repr(correct_text) != repr(text)

def normalize_eols(text, eol='\n'):
    if False:
        return 10
    "Use the same eol's in text"
    for (eol_char, _) in EOL_CHARS:
        if eol_char != eol:
            text = text.replace(eol_char, eol)
    return text

def fix_indentation(text, indent_chars):
    if False:
        for i in range(10):
            print('nop')
    'Replace tabs by spaces'
    return text.replace('\t', indent_chars)

def is_builtin(text):
    if False:
        for i in range(10):
            print('nop')
    'Test if passed string is the name of a Python builtin object'
    import builtins
    return text in [str(name) for name in dir(builtins) if not name.startswith('_')]

def is_keyword(text):
    if False:
        return 10
    'Test if passed string is the name of a Python keyword'
    import keyword
    return text in keyword.kwlist

def get_primary_at(source_code, offset, retry=True):
    if False:
        return 10
    "Return Python object in *source_code* at *offset*\n    Periods to the left of the cursor are carried forward\n      e.g. 'functools.par^tial' would yield 'functools.partial'\n    Retry prevents infinite recursion: retry only once\n    "
    obj = ''
    left = re.split('[^0-9a-zA-Z_.]', source_code[:offset])
    if left and left[-1]:
        obj = left[-1]
    right = re.split('\\W', source_code[offset:])
    if right and right[0]:
        obj += right[0]
    if obj and obj[0].isdigit():
        obj = ''
    if not obj and retry and offset and (source_code[offset - 1] in '([.'):
        return get_primary_at(source_code, offset - 1, retry=False)
    return obj

def split_source(source_code):
    if False:
        return 10
    'Split source code into lines\n    '
    eol_chars = get_eol_chars(source_code)
    if eol_chars:
        return source_code.split(eol_chars)
    else:
        return [source_code]

def get_identifiers(source_code):
    if False:
        return 10
    'Split source code into python identifier-like tokens'
    tokens = set(re.split('[^0-9a-zA-Z_.]', source_code))
    valid = re.compile('[a-zA-Z_]')
    return [token for token in tokens if re.match(valid, token)]

def path_components(path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the individual components of a given file path\n    string (for the local operating system).\n\n    Taken from https://stackoverflow.com/q/21498939/438386\n    '
    components = []
    while True:
        (new_path, tail) = os.path.split(path)
        components.append(tail)
        if new_path == path:
            break
        path = new_path
    components.append(new_path)
    components.reverse()
    return components

def differentiate_prefix(path_components0, path_components1):
    if False:
        print('Hello World!')
    '\n    Return the differentiated prefix of the given two iterables.\n\n    Taken from https://stackoverflow.com/q/21498939/438386\n    '
    longest_prefix = []
    root_comparison = False
    common_elmt = None
    for (index, (elmt0, elmt1)) in enumerate(zip(path_components0, path_components1)):
        if elmt0 != elmt1:
            if index == 2:
                root_comparison = True
            break
        else:
            common_elmt = elmt0
        longest_prefix.append(elmt0)
    file_name_length = len(path_components0[len(path_components0) - 1])
    path_0 = os.path.join(*path_components0)[:-file_name_length - 1]
    if len(longest_prefix) > 0:
        longest_path_prefix = os.path.join(*longest_prefix)
        longest_prefix_length = len(longest_path_prefix) + 1
        if path_0[longest_prefix_length:] != '' and (not root_comparison):
            path_0_components = path_components(path_0[longest_prefix_length:])
            if path_0_components[0] == '' and path_0_components[1] == '' and (len(path_0[longest_prefix_length:]) > 20):
                path_0_components.insert(2, common_elmt)
                path_0 = os.path.join(*path_0_components)
            else:
                path_0 = path_0[longest_prefix_length:]
        elif not root_comparison:
            path_0 = common_elmt
        elif sys.platform.startswith('linux') and path_0 == '':
            path_0 = '/'
    return path_0

def disambiguate_fname(files_path_list, filename):
    if False:
        print('Hello World!')
    'Get tab title without ambiguation.'
    fname = os.path.basename(filename)
    same_name_files = get_same_name_files(files_path_list, fname)
    if len(same_name_files) > 1:
        compare_path = shortest_path(same_name_files)
        if compare_path == filename:
            same_name_files.remove(path_components(filename))
            compare_path = shortest_path(same_name_files)
        diff_path = differentiate_prefix(path_components(filename), path_components(compare_path))
        diff_path_length = len(diff_path)
        path_component = path_components(diff_path)
        if diff_path_length > 20 and len(path_component) > 2:
            if path_component[0] != '/' and path_component[0] != '':
                path_component = [path_component[0], '...', path_component[-1]]
            else:
                path_component = [path_component[2], '...', path_component[-1]]
            diff_path = os.path.join(*path_component)
        fname = fname + ' - ' + diff_path
    return fname

def get_same_name_files(files_path_list, filename):
    if False:
        print('Hello World!')
    'Get a list of the path components of the files with the same name.'
    same_name_files = []
    for fname in files_path_list:
        if filename == os.path.basename(fname):
            same_name_files.append(path_components(fname))
    return same_name_files

def shortest_path(files_path_list):
    if False:
        print('Hello World!')
    'Shortest path between files in the list.'
    if len(files_path_list) > 0:
        shortest_path = files_path_list[0]
        shortest_path_length = len(files_path_list[0])
        for path_elmts in files_path_list:
            if len(path_elmts) < shortest_path_length:
                shortest_path_length = len(path_elmts)
                shortest_path = path_elmts
        return os.path.join(*shortest_path)

def camel_case_to_snake_case(input_str: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Convert a CamelCase string into a snake_case one.'
    return CAMEL_CASE_RE.sub('_', input_str).lower()