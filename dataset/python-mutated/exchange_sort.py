def exchange_sort(arr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reference : https://en.wikipedia.org/wiki/Sorting_algorithm#Exchange_sort\n    Complexity : O(n^2)\n    '
    arr_len = len(arr)
    for i in range(arr_len - 1):
        for j in range(i + 1, arr_len):
            if arr[i] > arr[j]:
                (arr[i], arr[j]) = (arr[j], arr[i])
    return arr