from __future__ import annotations
from ansible.module_utils import distro

class TestDistro:

    def test_info(self):
        if False:
            return 10
        info = distro.info()
        assert isinstance(info, dict), 'distro.info() returned %s (%s) which is not a dist' % (info, type(info))

    def test_id(self):
        if False:
            for i in range(10):
                print('nop')
        id = distro.id()
        assert isinstance(id, str), 'distro.id() returned %s (%s) which is not a string' % (id, type(id))