def insertion_sort(arr, simulation=False):
    if False:
        return 10
    ' Insertion Sort\n        Complexity: O(n^2)\n    '
    iteration = 0
    if simulation:
        print('iteration', iteration, ':', *arr)
    for i in range(len(arr)):
        cursor = arr[i]
        pos = i
        while pos > 0 and arr[pos - 1] > cursor:
            arr[pos] = arr[pos - 1]
            pos = pos - 1
        arr[pos] = cursor
        if simulation:
            iteration = iteration + 1
            print('iteration', iteration, ':', *arr)
    return arr