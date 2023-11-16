from __future__ import annotations

class TestModule:

    def tests(self):
        if False:
            return 10
        return {'world': lambda x: x.lower() == 'world'}