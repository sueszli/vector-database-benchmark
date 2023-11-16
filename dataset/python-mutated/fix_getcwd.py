u"""
Fixer for os.getcwd() -> os.getcwdu().
Also warns about "from os import getcwd", suggesting the above form.
"""
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name

class FixGetcwd(fixer_base.BaseFix):
    PATTERN = u"\n              power< 'os' trailer< dot='.' name='getcwd' > any* >\n              |\n              import_from< 'from' 'os' 'import' bad='getcwd' >\n              "

    def transform(self, node, results):
        if False:
            while True:
                i = 10
        if u'name' in results:
            name = results[u'name']
            name.replace(Name(u'getcwdu', prefix=name.prefix))
        elif u'bad' in results:
            self.cannot_convert(node, u'import os, use os.getcwd() instead.')
            return
        else:
            raise ValueError(u'For some reason, the pattern matcher failed.')