"""
Function to insert a new element at a given position

Takes the position of an array and an element as arguments
"""
arr = [1, 2, 3, 5]

def insert(arr, index, successor):
    if False:
        for i in range(10):
            print('nop')
    for index in range(index, len(arr)):
        (arr[index], successor) = (successor, arr[index])
    arr.append(successor)
print('Array before insertion:', arr)
insert(arr, 3, 4)
print('Array after insertion:', arr)