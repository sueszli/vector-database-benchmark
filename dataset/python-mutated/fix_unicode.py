"""Fixer for unicode.

* Changes unicode to str and unichr to chr.

* If "...\\u..." is not unicode literal change it into "...\\\\u...".

* Change u"..." into "...".

"""
from ..pgen2 import token
from .. import fixer_base
_mapping = {'unichr': 'chr', 'unicode': 'str'}

class FixUnicode(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "STRING | 'unicode' | 'unichr'"

    def start_tree(self, tree, filename):
        if False:
            while True:
                i = 10
        super(FixUnicode, self).start_tree(tree, filename)
        self.unicode_literals = 'unicode_literals' in tree.future_features

    def transform(self, node, results):
        if False:
            print('Hello World!')
        if node.type == token.NAME:
            new = node.clone()
            new.value = _mapping[node.value]
            return new
        elif node.type == token.STRING:
            val = node.value
            if not self.unicode_literals and val[0] in '\'"' and ('\\' in val):
                val = '\\\\'.join([v.replace('\\u', '\\\\u').replace('\\U', '\\\\U') for v in val.split('\\\\')])
            if val[0] in 'uU':
                val = val[1:]
            if val == node.value:
                return node
            new = node.clone()
            new.value = val
            return new