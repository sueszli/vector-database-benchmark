"""
UNFINISHED

Fixer for turning multiple lines like these:

    from __future__ import division
    from __future__ import absolute_import
    from __future__ import print_function

into a single line like this:

    from __future__ import (absolute_import, division, print_function)

This helps with testing of ``futurize``.
"""
from lib2to3 import fixer_base
from libfuturize.fixer_util import future_import

class FixOrderFutureImports(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = 'file_input'
    run_order = 10

    def transform(self, node, results):
        if False:
            while True:
                i = 10
        pass