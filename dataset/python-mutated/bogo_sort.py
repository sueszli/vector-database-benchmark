import random

def bogo_sort(arr, simulation=False):
    if False:
        i = 10
        return i + 15
    'Bogo Sort\n        Best Case Complexity: O(n)\n        Worst Case Complexity: O(∞)\n        Average Case Complexity: O(n(n-1)!)\n    '
    iteration = 0
    if simulation:
        print('iteration', iteration, ':', *arr)

    def is_sorted(arr):
        if False:
            return 10
        i = 0
        arr_len = len(arr)
        while i + 1 < arr_len:
            if arr[i] > arr[i + 1]:
                return False
            i += 1
        return True
    while not is_sorted(arr):
        random.shuffle(arr)
        if simulation:
            iteration = iteration + 1
            print('iteration', iteration, ':', *arr)
    return arr