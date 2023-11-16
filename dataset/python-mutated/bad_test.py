from __future__ import annotations

class TestModule:

    def tests(self):
        if False:
            for i in range(10):
                print('nop')
        raise TypeError('bad_test')