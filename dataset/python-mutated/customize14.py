"""Isolate Python 1.4- version-specific semantic actions here.
"""
from uncompyle6.semantics.consts import TABLE_DIRECT

def customize_for_version14(self, version):
    if False:
        for i in range(10):
            print('nop')
    TABLE_DIRECT.update({'print_expr_stmt': ('%|print %c\n', 0)})