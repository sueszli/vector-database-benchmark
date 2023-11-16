import unittest

class TestBits(unittest.TestCase):

    def test_pairwise_swap(self):
        if False:
            i = 10
            return i + 15
        bits = Bits()
        self.assertEqual(bits.pairwise_swap(0), 0)
        self.assertEqual(bits.pairwise_swap(1), 1)
        num = int('0000100111110110', base=2)
        expected = int('0000011011111001', base=2)
        self.assertEqual(bits.pairwise_swap(num), expected)
        self.assertRaises(TypeError, bits.pairwise_swap, None)
        print('Success: test_pairwise_swap')

def main():
    if False:
        return 10
    test = TestBits()
    test.test_pairwise_swap()
if __name__ == '__main__':
    main()