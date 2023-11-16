"""Isolate Python 2.5+ version-specific semantic actions here.
"""
from uncompyle6.semantics.consts import TABLE_DIRECT

def customize_for_version25(self, version):
    if False:
        while True:
            i = 10
    TABLE_DIRECT.update({'importmultiple': ('%|import %c%c\n', 2, 3), 'import_cont': (', %c', 2), 'with': ('%|with %c:\n%+%c%-', 0, 3), 'withasstmt': ('%|with %c as (%c):\n%+%c%-', 0, 2, 3)})

    def tryfinallystmt(node):
        if False:
            return 10
        if len(node[1][0]) == 1 and node[1][0][0] == 'stmt':
            if node[1][0][0][0] == 'try_except':
                node[1][0][0][0].kind = 'tf_try_except'
            if node[1][0][0][0] == 'tryelsestmt':
                node[1][0][0][0].kind = 'tf_tryelsestmt'
        self.default(node)
    self.n_tryfinallystmt = tryfinallystmt

    def n_import_from(node):
        if False:
            while True:
                i = 10
        if node[0].pattr > 0:
            node[2].pattr = '.' * node[0].pattr + node[2].pattr
        self.default(node)
    self.n_import_from = n_import_from