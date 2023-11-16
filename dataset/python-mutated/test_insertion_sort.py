import unittest

class TestInsertionSort(unittest.TestCase):

    def test_insertion_sort(self):
        if False:
            print('Hello World!')
        insertion_sort = InsertionSort()
        print('None input')
        self.assertRaises(TypeError, insertion_sort.sort, None)
        print('Empty input')
        self.assertEqual(insertion_sort.sort([]), [])
        print('One element')
        self.assertEqual(insertion_sort.sort([5]), [5])
        print('Two or more elements')
        data = [5, 1, 7, 2, 6, -3, 5, 7, -1]
        self.assertEqual(insertion_sort.sort(data), sorted(data))
        print('Success: test_insertion_sort')

def main():
    if False:
        i = 10
        return i + 15
    test = TestInsertionSort()
    test.test_insertion_sort()
if __name__ == '__main__':
    main()