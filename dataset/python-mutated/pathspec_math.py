from collections import namedtuple
from pathspec.util import normalize_file
from dvc.utils import relpath
PatternInfo = namedtuple('PatternInfo', ['patterns', 'file_info'])

def _not_ignore(rule):
    if False:
        i = 10
        return i + 15
    return (True, rule[1:]) if rule.startswith('!') else (False, rule)

def _is_comment(rule):
    if False:
        return 10
    return rule.startswith('#')

def _remove_slash(rule):
    if False:
        while True:
            i = 10
    if rule.startswith('\\'):
        return rule[1:]
    return rule

def _match_all_level(rule):
    if False:
        return 10
    if rule[:-1].find('/') >= 0 and (not rule.startswith('**/')):
        if rule.startswith('/'):
            rule = rule[1:]
        return (False, rule)
    if rule.startswith('**/'):
        rule = rule[3:]
    return (True, rule)

def change_rule(rule, rel):
    if False:
        for i in range(10):
            print('nop')
    rule = rule.strip()
    if _is_comment(rule):
        return rule
    (not_ignore, rule) = _not_ignore(rule)
    (match_all, rule) = _match_all_level(rule)
    rule = _remove_slash(rule)
    if not match_all:
        rule = f'/{rule}'
    else:
        rule = f'/**/{rule}'
    if not_ignore:
        rule = f'!/{rel}{rule}'
    else:
        rule = f'/{rel}{rule}'
    return normalize_file(rule)

def _change_dirname(dirname, pattern_list, new_dirname):
    if False:
        while True:
            i = 10
    if new_dirname == dirname:
        return pattern_list
    rel = relpath(dirname, new_dirname)
    if rel.startswith('..'):
        raise ValueError('change dirname can only change to parent path')
    return [PatternInfo(change_rule(rule.patterns, rel), rule.file_info) for rule in pattern_list]

def merge_patterns(flavour, pattern_a, prefix_a, pattern_b, prefix_b):
    if False:
        while True:
            i = 10
    '\n    Merge two path specification patterns.\n\n    This implementation merge two path specification patterns on different\n    bases. It returns the longest common parent directory, and the patterns\n    based on this new base directory.\n    '
    if not pattern_a:
        return (pattern_b, prefix_b)
    if not pattern_b:
        return (pattern_a, prefix_a)
    longest_common_dir = flavour.commonpath([prefix_a, prefix_b])
    new_pattern_a = _change_dirname(prefix_a, pattern_a, longest_common_dir)
    new_pattern_b = _change_dirname(prefix_b, pattern_b, longest_common_dir)
    if len(prefix_a) <= len(prefix_b):
        merged_pattern = new_pattern_a + new_pattern_b
    else:
        merged_pattern = new_pattern_b + new_pattern_a
    return (merged_pattern, longest_common_dir)