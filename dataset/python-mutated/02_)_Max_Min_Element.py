def minimum_number_of_comparison(arr, length):
    if False:
        i = 10
        return i + 15
    '\n    TIME COMPLEXITY : O(n)\n    if n is odd : number of comparison is 3(n - 1) / 2\n    if n is even : number of comparison is 3(n - 2) / 2 + 1\n    '
    if length % 2 == 0:
        maximum_number = max(arr[0], arr[1])
        minimum_number = min(arr[0], arr[1])
        i = 2
    else:
        maximum_number = minimum_number = arr[0]
        i = 1
    while i < length - 1:
        if arr[i] < arr[i + 1]:
            maximum_number = max(maximum_number, arr[i + 1])
            minimum_number = min(minimum_number, arr[i])
        else:
            maximum_number = max(maximum_number, arr[i])
            minimum_number = min(minimum_number, arr[i + 1])
        i += 2
    return (maximum_number, minimum_number)
arr = [1, 2, 3, 4]
(maximum, minimum) = minimum_number_of_comparison(arr, len(arr))
print(f'Maximum number among {arr} is {maximum}')
print(f'Minimum number among {arr} is {minimum}')