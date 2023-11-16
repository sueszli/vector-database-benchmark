from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .annotate import class_annotate, annotate, delay_type
from .type_spec import Type
from six import string_types as _string_types

@class_annotate()
class str:

    def __init__(self, v=''):
        if False:
            return 10
        self.val = v

    @classmethod
    def __type_info__(cls):
        if False:
            while True:
                i = 10
        return Type('str', python_class=cls)

    @annotate(delay_type.str, other=delay_type.str)
    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        assert isinstance(other, _string_types)
        return str(self.val + other.val)