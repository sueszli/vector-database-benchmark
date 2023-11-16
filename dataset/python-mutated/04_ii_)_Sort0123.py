def sort012_method2(arr):
    if False:
        print('Hello World!')
    '\n    Time Complexity : O(n) (Two traversals is required) (One traversal approach is discussed in 04_ii_)_Sort012.py)\n    Space Complexity : O(1)\n    '
    (zero, one, two) = (0, 0, 0)
    for number in arr:
        if number == 0:
            zero += 1
        elif number == 1:
            one += 1
        else:
            two += 1
    i = 0
    while zero > 0:
        arr[i] = 0
        i += 1
        zero -= 1
    while one > 0:
        arr[i] = 1
        i += 1
        one -= 1
    while two > 0:
        arr[i] = 2
        i += 1
        two -= 1
    return arr
print(sort012_method2([0, 2, 1, 2, 0]))