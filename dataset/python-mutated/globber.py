import re
_double_star_after_invalid_regex = re.compile('[^/\\\\]\\*\\*')
_double_star_first_before_invalid_regex = re.compile('^\\*\\*[^/]')
_double_star_middle_before_invalid_regex = re.compile('[^\\\\]\\*\\*[^/]')

def _match_component(pattern_component, file_name_component):
    if False:
        while True:
            i = 10
    if len(pattern_component) == 0 and len(file_name_component) == 0:
        return True
    elif len(pattern_component) == 0:
        return False
    elif len(file_name_component) == 0:
        return pattern_component == '*'
    elif pattern_component[0] == '*':
        return _match_component(pattern_component, file_name_component[1:]) or _match_component(pattern_component[1:], file_name_component)
    elif pattern_component[0] == '?':
        return _match_component(pattern_component[1:], file_name_component[1:])
    elif pattern_component[0] == '\\':
        return len(pattern_component) >= 2 and pattern_component[1] == file_name_component[0] and _match_component(pattern_component[2:], file_name_component[1:])
    elif pattern_component[0] != file_name_component[0]:
        return False
    else:
        return _match_component(pattern_component[1:], file_name_component[1:])

def _match_components(pattern_components, file_name_components):
    if False:
        return 10
    if len(pattern_components) == 0 and len(file_name_components) == 0:
        return True
    if len(pattern_components) == 0:
        return False
    if len(file_name_components) == 0:
        return len(pattern_components) == 1 and pattern_components[0] == '**'
    if pattern_components[0] == '**':
        return _match_components(pattern_components, file_name_components[1:]) or _match_components(pattern_components[1:], file_name_components)
    else:
        return _match_component(pattern_components[0], file_name_components[0]) and _match_components(pattern_components[1:], file_name_components[1:])

def match(pattern: str, file_name: str):
    if False:
        while True:
            i = 10
    "Match a glob pattern against a file name.\n\n    Glob pattern matching is for file names, which do not need to exist as\n    files on the file system.\n\n    A file name is a sequence of directory names, possibly followed by the name\n    of a file, with the components separated by a path separator. A glob\n    pattern is similar, except it may contain special characters: A '?' matches\n    any character in a name. A '*' matches any sequence of characters (possibly\n    empty) in a name. Both of these match only within a single component, i.e.,\n    they will not match a path separator. A component in a pattern may also be\n    a literal '**', which matches zero or more components in the complete file\n    name. A backslash '\\' in a pattern acts as an escape character, and\n    indicates that the following character is to be matched literally, even if\n    it is a special character.\n\n    Args:\n        pattern (str): The pattern to match. The path separator in patterns is\n                       always '/'.\n        file_name (str): The file name to match against. The path separator in\n                         file names is the platform separator\n\n    Returns:\n        bool: True if the pattern matches, False otherwise.\n    "
    if _double_star_after_invalid_regex.search(pattern) is not None or _double_star_first_before_invalid_regex.search(pattern) is not None or _double_star_middle_before_invalid_regex.search(pattern) is not None:
        raise ValueError('** in {} not alone between path separators'.format(pattern))
    pattern = pattern.rstrip('/')
    file_name = file_name.rstrip('/')
    while '**/**' in pattern:
        pattern = pattern.replace('**/**', '**')
    pattern_components = pattern.split('/')
    file_name_components = re.split('[\\\\/]', file_name)
    return _match_components(pattern_components, file_name_components)