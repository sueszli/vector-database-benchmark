"""
For the ``future`` package.

Turns any xrange calls into range calls and adds this import line:

    from builtins import range

at the top.
"""
from lib2to3.fixes.fix_xrange import FixXrange
from libfuturize.fixer_util import touch_import_top

class FixXrangeWithImport(FixXrange):

    def transform(self, node, results):
        if False:
            for i in range(10):
                print('nop')
        result = super(FixXrangeWithImport, self).transform(node, results)
        touch_import_top('builtins', 'range', node)
        return result