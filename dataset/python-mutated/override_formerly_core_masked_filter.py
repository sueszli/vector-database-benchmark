from __future__ import annotations

def override_formerly_core_masked_filter(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    return 'hello from overridden formerly_core_masked_filter'

class FilterModule(object):

    def filters(self):
        if False:
            while True:
                i = 10
        return {'formerly_core_masked_filter': override_formerly_core_masked_filter}