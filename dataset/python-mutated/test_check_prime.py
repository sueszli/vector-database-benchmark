import unittest

class TestMath(unittest.TestCase):

    def test_check_prime(self):
        if False:
            i = 10
            return i + 15
        math = Math()
        self.assertRaises(TypeError, math.check_prime, None)
        self.assertRaises(TypeError, math.check_prime, 98.6)
        self.assertEqual(math.check_prime(0), False)
        self.assertEqual(math.check_prime(1), False)
        self.assertEqual(math.check_prime(97), True)
        print('Success: test_check_prime')

def main():
    if False:
        while True:
            i = 10
    test = TestMath()
    test.test_check_prime()
if __name__ == '__main__':
    main()