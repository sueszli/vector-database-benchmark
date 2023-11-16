import unittest

class TestSelectionSort(unittest.TestCase):

    def test_selection_sort(self, func):
        if False:
            while True:
                i = 10
        print('None input')
        self.assertRaises(TypeError, func, None)
        print('Empty input')
        self.assertEqual(func([]), [])
        print('One element')
        self.assertEqual(func([5]), [5])
        print('Two or more elements')
        data = [5, 1, 7, 2, 6, -3, 5, 7, -10]
        self.assertEqual(func(data), sorted(data))
        print('Success: test_selection_sort\n')

def main():
    if False:
        print('Hello World!')
    test = TestSelectionSort()
    selection_sort = SelectionSort()
    test.test_selection_sort(selection_sort.sort)
    try:
        test.test_selection_sort(selection_sort.sort_recursive)
        test.test_selection_sort(selection_sort.sor_iterative_alt)
    except NameError:
        pass
if __name__ == '__main__':
    main()