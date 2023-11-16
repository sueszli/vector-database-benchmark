import unittest
from numba.tests.support import captured_stdout

class DocsTypedListUsageTest(unittest.TestCase):

    def test_ex_inferred_list_jit(self):
        if False:
            return 10
        with captured_stdout():
            from numba import njit
            from numba.typed import List

            @njit
            def foo():
                if False:
                    while True:
                        i = 10
                l = List()
                l.append(42)
                print(l[0])
                l[0] = 23
                print(l[0])
                print(len(l))
                l.pop()
                print(len(l))
                return l
            foo()

    def test_ex_inferred_list(self):
        if False:
            return 10
        with captured_stdout():
            from numba import njit
            from numba.typed import List

            @njit
            def foo(mylist):
                if False:
                    for i in range(10):
                        print('nop')
                for i in range(10, 20):
                    mylist.append(i)
                return mylist
            l = List()
            l.append(42)
            print(l[0])
            l[0] = 23
            print(l[0])
            print(len(l))
            l.pop()
            print(len(l))
            l = foo(l)
            print(len(l))
            py_list = [2, 3, 5]
            numba_list = List(py_list)
            print(len(numba_list))

    def test_ex_nested_list(self):
        if False:
            while True:
                i = 10
        with captured_stdout():
            from numba.typed import List
            mylist = List()
            for i in range(10):
                l = List()
                for i in range(10):
                    l.append(i)
                mylist.append(l)
            print(mylist)
if __name__ == '__main__':
    unittest.main()