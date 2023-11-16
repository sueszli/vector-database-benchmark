from __future__ import annotations

def test_subdir_filter(data):
    if False:
        while True:
            i = 10
    return '{0}_via_testfilter_from_subdir'.format(data)

class FilterModule(object):

    def filters(self):
        if False:
            return 10
        return {'test_subdir_filter': test_subdir_filter}