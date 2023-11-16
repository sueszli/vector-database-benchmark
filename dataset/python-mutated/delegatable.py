from __future__ import annotations
from ansible.playbook.attribute import FieldAttribute

class Delegatable:
    delegate_to = FieldAttribute(isa='string')
    delegate_facts = FieldAttribute(isa='bool')

    def _post_validate_delegate_to(self, attr, value, templar):
        if False:
            while True:
                i = 10
        'This method exists just to make it clear that ``Task.post_validate``\n        does not template this value, it is set via ``TaskExecutor._calculate_delegate_to``\n        '
        return value