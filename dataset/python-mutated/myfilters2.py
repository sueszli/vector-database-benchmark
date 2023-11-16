from __future__ import annotations

def testfilter2(data):
    if False:
        print('Hello World!')
    return '{0}_via_testfilter2_from_userdir'.format(data)

class FilterModule(object):

    def filters(self):
        if False:
            print('Hello World!')
        return {'testfilter2': testfilter2}