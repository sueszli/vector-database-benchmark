from __future__ import absolute_import, division, unicode_literals
from . import base
from collections import OrderedDict

def _attr_key(attr):
    if False:
        i = 10
        return i + 15
    "Return an appropriate key for an attribute for sorting\n\n    Attributes have a namespace that can be either ``None`` or a string. We\n    can't compare the two because they're different types, so we convert\n    ``None`` to an empty string first.\n\n    "
    return (attr[0][0] or '', attr[0][1])

class Filter(base.Filter):
    """Alphabetizes attributes for elements"""

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for token in base.Filter.__iter__(self):
            if token['type'] in ('StartTag', 'EmptyTag'):
                attrs = OrderedDict()
                for (name, value) in sorted(token['data'].items(), key=_attr_key):
                    attrs[name] = value
                token['data'] = attrs
            yield token