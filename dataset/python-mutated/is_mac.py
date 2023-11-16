from __future__ import annotations
from ansible.module_utils.common.network import is_mac

class TestModule(object):

    def tests(self):
        if False:
            while True:
                i = 10
        return {'is_mac': is_mac}