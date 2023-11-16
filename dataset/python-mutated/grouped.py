from __future__ import annotations
from ansible.utils.display import Display
display = Display()

def nochange(a):
    if False:
        i = 10
        return i + 15
    return a

def meaningoflife(a):
    if False:
        return 10
    return 42

class FilterModule(object):
    """ Ansible core jinja2 filters """

    def filters(self):
        if False:
            while True:
                i = 10
        return {'noop': nochange, 'ultimatequestion': meaningoflife, 'b64decode': nochange}