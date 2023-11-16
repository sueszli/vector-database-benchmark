"""
EXAMPLE:
    INPUT : [1, 2, -3, -5, -3, 1, 4, 6, -5, 3]
    OUTPUT : [-5, -3, -3, -5, 2, 1, 4, 6, 1, 3]
"""
'\n------------------------------------IMPORTANT NOTE------------------------------------------------\nI also solved this problem using Dutch National Flag (DNF) Algorithm \nsee 05_i_)_Move_All_Negative_Element_To_Front\n'

def negative_to_front(arr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Time Complexity : O(n)\n    Space Complexity : O(1)\n    '
    (pointer1, pointer2) = (0, len(arr) - 1)
    while pointer1 < pointer2:
        if arr[pointer1] >= 0 and arr[pointer2] >= 0:
            pointer2 -= 1
        elif arr[pointer1] < 0 and arr[pointer2] < 0:
            pointer1 += 1
        elif arr[pointer1] < 0 <= arr[pointer2]:
            pointer1 += 1
            pointer2 -= 1
        else:
            (arr[pointer1], arr[pointer2]) = (arr[pointer2], arr[pointer1])
    return arr
print(negative_to_front([1, 2, -3, -5, -3, 1, 4, 6, -5, 3]))