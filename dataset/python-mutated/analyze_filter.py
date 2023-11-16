import json
import logging
import re
from os import PathLike
from typing import Iterable, List, Tuple
from analyze.globber import match

def _match_path_and_rule(path: str, rule: str, patterns: Iterable[str]) -> bool:
    if False:
        while True:
            i = 10
    'Returns whether a given path matches a given rule.\n\n    Args:\n        path (str): A file path string.\n        rule (str): A rule file path string.\n        patterns (Iterable[str]): An iterable of pattern strings.\n\n    Returns:\n        bool: True if the path matches a rule. Otherwise, False.\n    '
    result = True
    for (s, fp, rp) in patterns:
        if match(rp, rule) and match(fp, path):
            result = s
    return result

def _parse_pattern(line: str) -> Tuple[str]:
    if False:
        i = 10
        return i + 15
    'Parses a given pattern line.\n\n    Args:\n        line (str): The line string that contains the rule.\n\n    Returns:\n        Tuple[str]: The parsed sign, file pattern, and rule pattern from the\n                    line.\n    '
    sep_char = ':'
    esc_char = '\\'
    file_pattern = ''
    rule_pattern = ''
    seen_separator = False
    sign = True
    u_line = line
    if line:
        if line[0] == '-':
            sign = False
            u_line = line[1:]
        elif line[0] == '+':
            u_line = line[1:]
    i = 0
    while i < len(u_line):
        c = u_line[i]
        i = i + 1
        if c == sep_char:
            if seen_separator:
                raise Exception('Invalid pattern: "' + line + '" Contains more than one separator!')
            seen_separator = True
            continue
        elif c == esc_char:
            next_c = u_line[i] if i < len(u_line) else None
            if next_c in ['+', '-', esc_char, sep_char]:
                i = i + 1
                c = next_c
        if seen_separator:
            rule_pattern = rule_pattern + c
        else:
            file_pattern = file_pattern + c
    if not rule_pattern:
        rule_pattern = '**'
    return (sign, file_pattern, rule_pattern)

def filter_sarif(input_sarif: PathLike, output_sarif: PathLike, patterns: List[str], split_lines: bool) -> None:
    if False:
        i = 10
        return i + 15
    'Filters a SARIF file with a given set of filter patterns.\n\n    Args:\n        input_sarif (PathLike): Input SARIF file path.\n        output_sarif (PathLike): Output SARIF file path.\n        patterns (PathLike): List of filter pattern strings.\n        split_lines (PathLike): Whether to split lines in individual patterns.\n    '
    if split_lines:
        tmp = []
        for p in patterns:
            tmp = tmp + re.split('\r?\n', p)
        patterns = tmp
    patterns = [_parse_pattern(p) for p in patterns if p]
    logging.debug('Given patterns:')
    for (s, fp, rp) in patterns:
        logging.debug('files: {file_pattern}    rules: {rule_pattern} ({sign})'.format(file_pattern=fp, rule_pattern=rp, sign='positive' if s else 'negative'))
    with open(input_sarif, 'r') as f:
        s = json.load(f)
    for run in s.get('runs', []):
        if run.get('results', []):
            new_results = []
            for r in run['results']:
                if r.get('locations', []):
                    new_locations = []
                    for l in r['locations']:
                        uri = l.get('physicalLocation', {}).get('artifactLocation', {}).get('uri', None)
                        ruleId = r['ruleId']
                        if uri is None or _match_path_and_rule(uri, ruleId, patterns):
                            new_locations.append(l)
                    r['locations'] = new_locations
                    if new_locations:
                        new_results.append(r)
                else:
                    new_results.append(r)
            run['results'] = new_results
    with open(output_sarif, 'w') as f:
        json.dump(s, f, indent=2)