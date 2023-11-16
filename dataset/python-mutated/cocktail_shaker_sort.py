def cocktail_shaker_sort(arr):
    if False:
        print('Hello World!')
    '\n    Cocktail_shaker_sort\n    Sorting a given array\n    mutation of bubble sort\n\n    reference: https://en.wikipedia.org/wiki/Cocktail_shaker_sort\n    \n    Worst-case performance: O(N^2)\n    '

    def swap(i, j):
        if False:
            return 10
        (arr[i], arr[j]) = (arr[j], arr[i])
    n = len(arr)
    swapped = True
    while swapped:
        swapped = False
        for i in range(1, n):
            if arr[i - 1] > arr[i]:
                swap(i - 1, i)
                swapped = True
        if swapped == False:
            return arr
        swapped = False
        for i in range(n - 1, 0, -1):
            if arr[i - 1] > arr[i]:
                swap(i - 1, i)
                swapped = True
    return arr