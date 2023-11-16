from __future__ import annotations

def testfilter(data):
    if False:
        while True:
            i = 10
    return '{0}_via_testfilter_from_userdir'.format(data)

class FilterModule(object):

    def filters(self):
        if False:
            i = 10
            return i + 15
        return {'testfilter': testfilter}