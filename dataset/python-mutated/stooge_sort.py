"""

Stooge Sort
Time Complexity : O(n2.709)
Reference: https://www.geeksforgeeks.org/stooge-sort/

"""

def stoogesort(arr, l, h):
    if False:
        while True:
            i = 10
    if l >= h:
        return
    if arr[l] > arr[h]:
        t = arr[l]
        arr[l] = arr[h]
        arr[h] = t
    if h - l + 1 > 2:
        t = int((h - l + 1) / 3)
        stoogesort(arr, l, h - t)
        stoogesort(arr, l + t, h)
        stoogesort(arr, l, h - t)
if __name__ == '__main__':
    array = [1, 3, 64, 5, 7, 8]
    n = len(array)
    stoogesort(array, 0, n - 1)
    for i in range(0, n):
        print(array[i], end=' ')