def kth_min_element(arr, k):
    if False:
        for i in range(10):
            print('nop')
    '\n    Time complexity : O(nlogn)\n    Space Complexity : O(1)\n    '
    arr.sort()
    return arr[k - 1]
print(kth_min_element([1, 2, 3, 4, 5], 3))