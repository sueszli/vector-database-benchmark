import unittest

class TestRadixSort(unittest.TestCase):

    def test_sort(self):
        if False:
            print('Hello World!')
        radix_sort = RadixSort()
        self.assertRaises(TypeError, radix_sort.sort, None)
        self.assertEqual(radix_sort.sort([]), [])
        array = [128, 256, 164, 8, 2, 148, 212, 242, 244]
        expected = [2, 8, 128, 148, 164, 212, 242, 244, 256]
        self.assertEqual(radix_sort.sort(array), expected)
        print('Success: test_sort')

def main():
    if False:
        return 10
    test = TestRadixSort()
    test.test_sort()
if __name__ == '__main__':
    main()