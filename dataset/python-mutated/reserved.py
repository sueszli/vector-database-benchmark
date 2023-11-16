from __future__ import annotations
from ansible.playbook import Play
from ansible.playbook.block import Block
from ansible.playbook.role import Role
from ansible.playbook.task import Task
from ansible.utils.display import Display
display = Display()

def get_reserved_names(include_private=True):
    if False:
        return 10
    ' this function returns the list of reserved names associated with play objects'
    public = set()
    private = set()
    result = set()
    class_list = [Play, Role, Block, Task]
    for aclass in class_list:
        for (name, attr) in aclass.fattributes.items():
            if attr.private:
                private.add(name)
            else:
                public.add(name)
    if 'action' in public:
        public.add('local_action')
    if 'loop' in private or 'loop' in public:
        public.add('with_')
    if include_private:
        result = public.union(private)
    else:
        result = public
    return result

def warn_if_reserved(myvars, additional=None):
    if False:
        while True:
            i = 10
    ' this function warns if any variable passed conflicts with internally reserved names '
    if additional is None:
        reserved = _RESERVED_NAMES
    else:
        reserved = _RESERVED_NAMES.union(additional)
    varnames = set(myvars)
    varnames.discard('vars')
    for varname in varnames.intersection(reserved):
        display.warning('Found variable using reserved name: %s' % varname)

def is_reserved_name(name):
    if False:
        print('Hello World!')
    return name in _RESERVED_NAMES
_RESERVED_NAMES = frozenset(get_reserved_names())