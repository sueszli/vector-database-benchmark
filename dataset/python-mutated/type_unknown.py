from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .type_spec import Type

class unknown:
    """
    unknown is basically Any type.
    """

    @classmethod
    def __type_info__(cls):
        if False:
            print('Hello World!')
        return Type('unknown', python_class=cls)

    def __init__(self, val=None):
        if False:
            print('Hello World!')
        self.val = val