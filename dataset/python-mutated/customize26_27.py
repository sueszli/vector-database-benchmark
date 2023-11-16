"""Isolate Python 2.6 and 2.7 version-specific semantic actions here.
"""
from uncompyle6.semantics.consts import TABLE_DIRECT

def customize_for_version26_27(self, version):
    if False:
        while True:
            i = 10
    if version > (2, 6):
        TABLE_DIRECT.update({'except_cond2': ('%|except %c as %c:\n', 1, 5), 'call_generator': ('%c%P', 0, (1, -1, ', ', 100))})
    else:
        TABLE_DIRECT.update({'testtrue_then': ('not %p', (0, 22))})

    def n_call(node):
        if False:
            while True:
                i = 10
        mapping = self._get_mapping(node)
        key = node
        for i in mapping[1:]:
            key = key[i]
            pass
        if key.kind == 'CALL_FUNCTION_1':
            args_node = node[-2]
            if args_node == 'expr':
                n = args_node[0]
                if n == 'generator_exp':
                    node.kind = 'call_generator'
                    pass
                pass
        self.default(node)
    self.n_call = n_call

    def n_import_from(node):
        if False:
            print('Hello World!')
        if node[0].pattr > 0:
            node[2].pattr = '.' * node[0].pattr + node[2].pattr
        self.default(node)
    self.n_import_from = n_import_from