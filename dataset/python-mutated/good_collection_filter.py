from __future__ import annotations

class FilterModule:

    def filters(self):
        if False:
            for i in range(10):
                print('nop')
        return {'hello': lambda x: 'Hello, %s!' % x}