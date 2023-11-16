"""
EXAMPLE:
    INPUT : [1, 2, -3, -5, -3, 1, 4, 6, -5, 3]
    OUTPUT : [-5, -3, -3, -5, 2, 1, 4, 6, 1, 3]
"""
'\n------------------------------------IMPORTANT NOTE------------------------------------------------\nI also solved this problem using 2 pointer approach see 05_ii_)_Move_All_Negative_Element_To_Front\n'

def negative_to_front(arr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Time Complexity : O(n)\n    Space Complexity : O(1)\n    '
    (low, mid) = (0, 0)
    while mid < len(arr):
        if arr[mid] < 0:
            (arr[mid], arr[low]) = (arr[low], arr[mid])
            mid += 1
            low += 1
        else:
            mid += 1
    return arr
print(negative_to_front([1, 23, -4, -5, 3, -4, -2, 1]))