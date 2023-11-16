"""

Gnome Sort
Best case performance is O(n)
Worst case performance is O(n^2)

"""

def gnome_sort(arr):
    if False:
        while True:
            i = 10
    n = len(arr)
    index = 0
    while index < n:
        if index == 0 or arr[index] >= arr[index - 1]:
            index = index + 1
        else:
            (arr[index], arr[index - 1]) = (arr[index - 1], arr[index])
            index = index - 1
    return arr