import unittest

class TestBit(unittest.TestCase):

    def test_insert_m_into_n(self):
        if False:
            for i in range(10):
                print('nop')
        n = int('0000010000111101', base=2)
        m = int('0000000000010011', base=2)
        expected = int('0000010001001101', base=2)
        bits = Bits()
        self.assertEqual(bits.insert_m_into_n(m, n, i=2, j=6), expected)
        print('Success: test_insert_m_into_n')

def main():
    if False:
        i = 10
        return i + 15
    test = TestBit()
    test.test_insert_m_into_n()
if __name__ == '__main__':
    main()