def quick_sort(arr, simulation=False):
    if False:
        for i in range(10):
            print('nop')
    ' Quick sort\n        Complexity: best O(n log(n)) avg O(n log(n)), worst O(N^2)\n    '
    iteration = 0
    if simulation:
        print('iteration', iteration, ':', *arr)
    (arr, _) = quick_sort_recur(arr, 0, len(arr) - 1, iteration, simulation)
    return arr

def quick_sort_recur(arr, first, last, iteration, simulation):
    if False:
        return 10
    if first < last:
        pos = partition(arr, first, last)
        if simulation:
            iteration = iteration + 1
            print('iteration', iteration, ':', *arr)
        (_, iteration) = quick_sort_recur(arr, first, pos - 1, iteration, simulation)
        (_, iteration) = quick_sort_recur(arr, pos + 1, last, iteration, simulation)
    return (arr, iteration)

def partition(arr, first, last):
    if False:
        while True:
            i = 10
    wall = first
    for pos in range(first, last):
        if arr[pos] < arr[last]:
            (arr[pos], arr[wall]) = (arr[wall], arr[pos])
            wall += 1
    (arr[wall], arr[last]) = (arr[last], arr[wall])
    return wall