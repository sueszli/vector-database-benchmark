def selection_sort(arr, simulation=False):
    if False:
        return 10
    ' Selection Sort\n        Complexity: O(n^2)\n    '
    iteration = 0
    if simulation:
        print('iteration', iteration, ':', *arr)
    for i in range(len(arr)):
        minimum = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[minimum]:
                minimum = j
        (arr[minimum], arr[i]) = (arr[i], arr[minimum])
        if simulation:
            iteration = iteration + 1
            print('iteration', iteration, ':', *arr)
    return arr