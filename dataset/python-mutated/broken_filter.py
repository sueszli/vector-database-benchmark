from __future__ import annotations

class FilterModule(object):

    def filters(self):
        if False:
            return 10
        return {'broken': lambda x: 'broken'}
raise Exception('This is a broken filter plugin.')