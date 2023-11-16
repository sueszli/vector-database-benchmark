import unittest

class TestLongestCommonSubseq(unittest.TestCase):

    def test_longest_common_subseq(self):
        if False:
            i = 10
            return i + 15
        str_comp = StringCompare()
        self.assertRaises(TypeError, str_comp.longest_common_subseq, None, None)
        self.assertEqual(str_comp.longest_common_subseq('', ''), '')
        str0 = 'ABCDEFGHIJ'
        str1 = 'FOOBCDBCDE'
        expected = 'BCDE'
        self.assertEqual(str_comp.longest_common_subseq(str0, str1), expected)
        print('Success: test_longest_common_subseq')

def main():
    if False:
        while True:
            i = 10
    test = TestLongestCommonSubseq()
    test.test_longest_common_subseq()
if __name__ == '__main__':
    main()