import unittest
import win32net
import win32netcon

class TestCase(unittest.TestCase):

    def testGroupsGoodResume(self, server=None):
        if False:
            i = 10
            return i + 15
        res = 0
        level = 0
        while True:
            (user_list, total, res) = win32net.NetGroupEnum(server, level, res)
            for i in user_list:
                pass
            if not res:
                break

    def testGroupsBadResume(self, server=None):
        if False:
            for i in range(10):
                print('nop')
        res = 1
        self.assertRaises(win32net.error, win32net.NetGroupEnum, server, 0, res)
if __name__ == '__main__':
    unittest.main()