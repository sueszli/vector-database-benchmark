from __future__ import annotations


def testtest(data):
    return data == 'from_user'


class TestModule(object):
    def tests(self):
        return {
            'testtest': testtest
        }
