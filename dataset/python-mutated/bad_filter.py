from __future__ import annotations

class FilterModule:

    def filters(self):
        if False:
            i = 10
            return i + 15
        raise TypeError('bad_filter')