def cycle_sort(arr):
    if False:
        for i in range(10):
            print('nop')
    '\n    cycle_sort\n    This is based on the idea that the permutations to be sorted\n    can be decomposed into cycles,\n    and the results can be individually sorted by cycling.\n    \n    reference: https://en.wikipedia.org/wiki/Cycle_sort\n    \n    Average time complexity : O(N^2)\n    Worst case time complexity : O(N^2)\n    '
    len_arr = len(arr)
    for cur in range(len_arr - 1):
        item = arr[cur]
        index = cur
        for i in range(cur + 1, len_arr):
            if arr[i] < item:
                index += 1
        if index == cur:
            continue
        while item == arr[index]:
            index += 1
        (arr[index], item) = (item, arr[index])
        while index != cur:
            index = cur
            for i in range(cur + 1, len_arr):
                if arr[i] < item:
                    index += 1
            while item == arr[index]:
                index += 1
            (arr[index], item) = (item, arr[index])
    return arr