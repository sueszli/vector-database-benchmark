def search(arr, element):
    if False:
        for i in range(10):
            print('nop')
    for index in range(0, len(arr)):
        if arr[index] == element:
            return index
    return -1
arr = [20, 5, 7, 25]
element = 5
print('Array Searched:', arr)
print('Searching for:', element)
print('Searched index=', search(arr, element))