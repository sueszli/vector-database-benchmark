from __future__ import annotations

def testtest(data):
    if False:
        return 10
    return data == 'from_user2'

class TestModule(object):

    def tests(self):
        if False:
            for i in range(10):
                print('nop')
        return {'testtest2': testtest}