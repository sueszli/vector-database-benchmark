def bucket_sort(arr):
    if False:
        return 10
    ' Bucket Sort\n        Complexity: O(n^2)\n        The complexity is dominated by nextSort\n    '
    num_buckets = len(arr)
    buckets = [[] for bucket in range(num_buckets)]
    for value in arr:
        index = value * num_buckets // (max(arr) + 1)
        buckets[index].append(value)
    sorted_list = []
    for i in range(num_buckets):
        sorted_list.extend(next_sort(buckets[i]))
    return sorted_list

def next_sort(arr):
    if False:
        while True:
            i = 10
    for i in range(1, len(arr)):
        j = i - 1
        key = arr[i]
        while arr[j] > key and j >= 0:
            arr[j + 1] = arr[j]
            j = j - 1
        arr[j + 1] = key
    return arr