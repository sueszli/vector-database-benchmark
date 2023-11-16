def merge_sort(arr):
    if False:
        print('Hello World!')
    ' Merge Sort\n        Complexity: O(n log(n))\n    '
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    (left, right) = (merge_sort(arr[:mid]), merge_sort(arr[mid:]))
    merge(left, right, arr)
    return arr

def merge(left, right, merged):
    if False:
        print('Hello World!')
    ' Merge helper\n        Complexity: O(n)\n    '
    (left_cursor, right_cursor) = (0, 0)
    while left_cursor < len(left) and right_cursor < len(right):
        if left[left_cursor] <= right[right_cursor]:
            merged[left_cursor + right_cursor] = left[left_cursor]
            left_cursor += 1
        else:
            merged[left_cursor + right_cursor] = right[right_cursor]
            right_cursor += 1
    for left_cursor in range(left_cursor, len(left)):
        merged[left_cursor + right_cursor] = left[left_cursor]
    for right_cursor in range(right_cursor, len(right)):
        merged[left_cursor + right_cursor] = right[right_cursor]