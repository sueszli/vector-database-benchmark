from __future__ import annotations

def override_formerly_core_masked_test(value, *args, **kwargs):
    if False:
        while True:
            i = 10
    if value != 'hello override':
        raise Exception('expected "hello override" only...')
    return True

class TestModule(object):

    def tests(self):
        if False:
            i = 10
            return i + 15
        return {'formerly_core_masked_test': override_formerly_core_masked_test}