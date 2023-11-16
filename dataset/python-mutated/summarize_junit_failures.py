from __future__ import annotations
import os
import re
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
TEXT_RED = '\x1b[31m'
TEXT_YELLOW = '\x1b[33m'
TEXT_RESET = '\x1b[0m'

@lru_cache(maxsize=None)
def translate_classname(classname):
    if False:
        for i in range(10):
            print('nop')
    if not classname:
        return None
    context = Path.cwd()
    parts = classname.split('.')
    for (offset, component) in enumerate(parts, 1):
        candidate = context / component
        if candidate.is_dir():
            context = candidate
        else:
            candidate = context / (component + '.py')
            if candidate.is_file():
                context = candidate
            break
    parts = parts[offset:]
    val = str(context.relative_to(Path.cwd()))
    if parts:
        val += '::' + '.'.join(parts)
    return val

def translate_name(testcase):
    if False:
        print('Hello World!')
    classname = translate_classname(testcase.get('classname'))
    name = testcase.get('name')
    if not classname:
        return translate_classname(name)
    return f'{classname}::{name}'

def summarize_file(input, test_type, backend):
    if False:
        return 10
    root = ET.parse(input)
    testsuite = root.find('.//testsuite')
    fail_message_parts = []
    num_failures = int(testsuite.get('failures'))
    if num_failures:
        fail_message_parts.append(f"{num_failures} failure{('' if num_failures == 1 else 's')}")
    num_errors = int(testsuite.get('errors'))
    if num_errors:
        fail_message_parts.append(f"{num_errors} error{('' if num_errors == 1 else 's')}")
    if not fail_message_parts:
        return
    print(f"\n{TEXT_RED}==== {test_type} {backend}: {', '.join(fail_message_parts)} ===={TEXT_RESET}\n")
    for testcase in testsuite.findall('.//testcase[error]'):
        case_name = translate_name(testcase)
        for err in testcase.iterfind('error'):
            print(f"{case_name}: {TEXT_YELLOW}{err.get('message')}{TEXT_RESET}")
    for testcase in testsuite.findall('.//testcase[failure]'):
        case_name = translate_name(testcase)
        for failure in testcase.iterfind('failure'):
            print(f"{case_name}: {TEXT_YELLOW}{failure.get('message')}{TEXT_RESET}")
if __name__ == '__main__':
    fname_pattern = re.compile('^test_result-(?P<test_type>.*?)-(?P<backend>.*).xml$')
    for fname in sys.argv[1:]:
        match = fname_pattern.match(os.path.basename(fname))
        if not match:
            exit(f'I cannot understand the name format of {fname!r}')
        with open(fname) as fh:
            summarize_file(fh, **match.groupdict())