from __future__ import annotations
from ansible.utils.display import Display
display = Display()

def nochange(a):
    if False:
        for i in range(10):
            print('nop')
    return a

class FilterModule(object):
    """ Ansible core jinja2 filters """

    def filters(self):
        if False:
            while True:
                i = 10
        return {'noop': nochange, 'nested': nochange}