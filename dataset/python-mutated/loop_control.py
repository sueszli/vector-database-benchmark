from __future__ import annotations
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import FieldAttributeBase

class LoopControl(FieldAttributeBase):
    loop_var = NonInheritableFieldAttribute(isa='string', default='item', always_post_validate=True)
    index_var = NonInheritableFieldAttribute(isa='string', always_post_validate=True)
    label = NonInheritableFieldAttribute(isa='string')
    pause = NonInheritableFieldAttribute(isa='float', default=0, always_post_validate=True)
    extended = NonInheritableFieldAttribute(isa='bool', always_post_validate=True)
    extended_allitems = NonInheritableFieldAttribute(isa='bool', default=True, always_post_validate=True)

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(LoopControl, self).__init__()

    @staticmethod
    def load(data, variable_manager=None, loader=None):
        if False:
            i = 10
            return i + 15
        t = LoopControl()
        return t.load_data(data, variable_manager=variable_manager, loader=loader)