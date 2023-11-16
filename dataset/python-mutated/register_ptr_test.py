import unittest
from register_ptr import *

class RegisterPtrTest(unittest.TestCase):

    def testIt(self):
        if False:
            print('Hello World!')

        class B(A):

            def f(self):
                if False:
                    print('Hello World!')
                return 10
        a = New()
        b = B()
        self.assertEqual(Call(a), 0)
        self.assertEqual(Call(b), 10)

        def fails():
            if False:
                print('Hello World!')
            Fail(A())
        self.assertRaises(TypeError, fails)
        self.assertEqual(Fail(a), 0)
if __name__ == '__main__':
    unittest.main()