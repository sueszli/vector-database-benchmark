from __future__ import annotations

class FilterModule:

    def filters(self):
        if False:
            for i in range(10):
                print('nop')
        raise TypeError('bad_collection_filter')