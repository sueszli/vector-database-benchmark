from __future__ import unicode_literals, division, absolute_import, print_function
from powerline.lib.unicode import unicode

def havemarks(*args, **kwargs):
    if False:
        while True:
            i = 10
    origin = kwargs.get('origin', '')
    for (i, v) in enumerate(args):
        if not hasattr(v, 'mark'):
            raise AssertionError('Value #{0}/{1} ({2!r}) has no attribute `mark`'.format(origin, i, v))
        if isinstance(v, dict):
            for (key, val) in v.items():
                havemarks(key, val, origin=origin + '[' + unicode(i) + ']/' + unicode(key))
        elif isinstance(v, list):
            havemarks(*v, origin=origin + '[' + unicode(i) + ']')