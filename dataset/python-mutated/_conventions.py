"""Central location for shared argparse convention definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import functools
from absl import app as absl_app
from absl import flags
_help_wrap = functools.partial(flags.text_wrap, length=80, indent='', firstline_indent='\n')

def _stdout_utf8():
    if False:
        i = 10
        return i + 15
    try:
        codecs.lookup('utf-8')
    except LookupError:
        return False
    return sys.stdout.encoding == 'UTF-8'
if _stdout_utf8():
    help_wrap = _help_wrap
else:

    def help_wrap(text, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return _help_wrap(text, *args, **kwargs).replace(u'\ufeff', u'')
absl_app.HelpshortFlag.SHORT_NAME = 'h'