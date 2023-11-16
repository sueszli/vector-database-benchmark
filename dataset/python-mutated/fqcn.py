from __future__ import annotations

def add_internal_fqcns(names):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a sequence of action/module names, returns a list of these names\n    with the same names with the prefixes `ansible.builtin.` and\n    `ansible.legacy.` added for all names that are not already FQCNs.\n    '
    result = []
    for name in names:
        result.append(name)
        if '.' not in name:
            result.append('ansible.builtin.%s' % name)
            result.append('ansible.legacy.%s' % name)
    return result