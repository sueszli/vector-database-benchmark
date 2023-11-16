from __future__ import annotations

class TestModule:

    def tests(self):
        if False:
            for i in range(10):
                print('nop')
        return {'world': lambda x: x.lower() == 'world'}