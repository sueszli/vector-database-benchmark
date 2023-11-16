def pancake_sort(arr):
    if False:
        while True:
            i = 10
    '\n    Pancake_sort\n    Sorting a given array\n    mutation of selection sort\n\n    reference: https://www.geeksforgeeks.org/pancake-sorting/\n    \n    Overall time complexity : O(N^2)\n    '
    len_arr = len(arr)
    if len_arr <= 1:
        return arr
    for cur in range(len(arr), 1, -1):
        index_max = arr.index(max(arr[0:cur]))
        if index_max + 1 != cur:
            if index_max != 0:
                arr[:index_max + 1] = reversed(arr[:index_max + 1])
            arr[:cur] = reversed(arr[:cur])
    return arr