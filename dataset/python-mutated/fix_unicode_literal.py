from lib2to3 import fixer_base
from lib2to3.fixer_util import String

class FixUnicodeLiteral(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "\n    power< 'u'\n        trailer<\n            '('\n                arg=any\n            ')'\n        >\n    >\n    "

    def transform(self, node, results):
        if False:
            return 10
        arg = results['arg']
        node.replace(String('u' + arg.value, prefix=node.prefix))