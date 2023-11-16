from collections import UserList
from test import list_tests
import unittest

class UserListTest(list_tests.CommonTest):
    type2test = UserList

    def test_getslice(self):
        if False:
            i = 10
            return i + 15
        super().test_getslice()
        l = [0, 1, 2, 3, 4]
        u = self.type2test(l)
        for i in range(-3, 6):
            self.assertEqual(u[:i], l[:i])
            self.assertEqual(u[i:], l[i:])
            for j in range(-3, 6):
                self.assertEqual(u[i:j], l[i:j])

    def test_slice_type(self):
        if False:
            while True:
                i = 10
        l = [0, 1, 2, 3, 4]
        u = UserList(l)
        self.assertIsInstance(u[:], u.__class__)
        self.assertEqual(u[:], u)

    def test_add_specials(self):
        if False:
            for i in range(10):
                print('nop')
        u = UserList('spam')
        u2 = u + 'eggs'
        self.assertEqual(u2, list('spameggs'))

    def test_radd_specials(self):
        if False:
            return 10
        u = UserList('eggs')
        u2 = 'spam' + u
        self.assertEqual(u2, list('spameggs'))
        u2 = u.__radd__(UserList('spam'))
        self.assertEqual(u2, list('spameggs'))

    def test_iadd(self):
        if False:
            while True:
                i = 10
        super().test_iadd()
        u = [0, 1]
        u += UserList([0, 1])
        self.assertEqual(u, [0, 1, 0, 1])

    def test_mixedcmp(self):
        if False:
            return 10
        u = self.type2test([0, 1])
        self.assertEqual(u, [0, 1])
        self.assertNotEqual(u, [0])
        self.assertNotEqual(u, [0, 2])

    def test_mixedadd(self):
        if False:
            for i in range(10):
                print('nop')
        u = self.type2test([0, 1])
        self.assertEqual(u + [], u)
        self.assertEqual(u + [2], [0, 1, 2])

    def test_getitemoverwriteiter(self):
        if False:
            return 10

        class T(self.type2test):

            def __getitem__(self, key):
                if False:
                    while True:
                        i = 10
                return str(key) + '!!!'
        self.assertEqual(next(iter(T((1, 2)))), '0!!!')

    def test_userlist_copy(self):
        if False:
            print('Hello World!')
        u = self.type2test([6, 8, 1, 9, 1])
        v = u.copy()
        self.assertEqual(u, v)
        self.assertEqual(type(u), type(v))
if __name__ == '__main__':
    unittest.main()