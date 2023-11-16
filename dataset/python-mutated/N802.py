import unittest

def badAllowed():
    if False:
        return 10
    pass

def stillBad():
    if False:
        for i in range(10):
            print('nop')
    pass

class Test(unittest.TestCase):

    def badAllowed(self):
        if False:
            while True:
                i = 10
        return super().tearDown()

    def stillBad(self):
        if False:
            i = 10
            return i + 15
        return super().tearDown()