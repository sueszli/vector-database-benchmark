import os
import random
from viztracer import VizTracer

def merge_sort(collection):
    if False:
        print('Hello World!')
    'Pure implementation of the merge sort algorithm in Python\n\n    :param collection: some mutable ordered collection with heterogeneous\n    comparable items inside\n    :return: the same collection ordered by ascending\n\n    Examples:\n    >>> merge_sort([0, 5, 3, 2, 2])\n    [0, 2, 2, 3, 5]\n\n    >>> merge_sort([])\n    []\n\n    >>> merge_sort([-2, -5, -45])\n    [-45, -5, -2]\n    '

    def merge(left, right):
        if False:
            return 10
        'merge left and right\n        :param left: left collection\n        :param right: right collection\n        :return: merge result\n        '
        result = []
        while left and right:
            result.append((left if left[0] <= right[0] else right).pop(0))
        return result + left + right
    if len(collection) <= 1:
        return collection
    mid = len(collection) // 2
    return merge(merge_sort(collection[:mid]), merge_sort(collection[mid:]))

def quick_sort(collection):
    if False:
        for i in range(10):
            print('nop')
    'Pure implementation of quick sort algorithm in Python\n\n    :param collection: some mutable ordered collection with heterogeneous\n    comparable items inside\n    :return: the same collection ordered by ascending\n\n    Examples:\n    >>> quick_sort([0, 5, 3, 2, 2])\n    [0, 2, 2, 3, 5]\n\n    >>> quick_sort([])\n    []\n\n    >>> quick_sort([-2, -5, -45])\n    [-45, -5, -2]\n    '
    length = len(collection)
    if length <= 1:
        return collection
    else:
        pivot = collection.pop()
        (greater, lesser) = ([], [])
        for element in collection:
            if element > pivot:
                greater.append(element)
            else:
                lesser.append(element)
        return quick_sort(lesser) + [pivot] + quick_sort(greater)

def heapify(unsorted, index, heap_size):
    if False:
        while True:
            i = 10
    largest = index
    left_index = 2 * index + 1
    right_index = 2 * index + 2
    if left_index < heap_size and unsorted[left_index] > unsorted[largest]:
        largest = left_index
    if right_index < heap_size and unsorted[right_index] > unsorted[largest]:
        largest = right_index
    if largest != index:
        (unsorted[largest], unsorted[index]) = (unsorted[index], unsorted[largest])
        heapify(unsorted, largest, heap_size)

def heap_sort(unsorted):
    if False:
        return 10
    '\n    Pure implementation of the heap sort algorithm in Python\n    :param collection: some mutable ordered collection with heterogeneous\n    comparable items inside\n    :return: the same collection ordered by ascending\n\n    Examples:\n    >>> heap_sort([0, 5, 3, 2, 2])\n    [0, 2, 2, 3, 5]\n\n    >>> heap_sort([])\n    []\n\n    >>> heap_sort([-2, -5, -45])\n    [-45, -5, -2]\n    '
    n = len(unsorted)
    for i in range(n // 2 - 1, -1, -1):
        heapify(unsorted, i, n)
    for i in range(n - 1, 0, -1):
        (unsorted[0], unsorted[i]) = (unsorted[i], unsorted[0])
        heapify(unsorted, 0, i)
    return unsorted
arr1 = [random.randrange(100000) for _ in range(500)]
arr2 = [random.randrange(100000) for _ in range(500)]
arr3 = [random.randrange(100000) for _ in range(500)]
with VizTracer(output_file=os.path.join(os.path.dirname(__file__), '../', 'json/different_sorts.json'), file_info=True) as _:
    merge_sort(arr1)
    quick_sort(arr2)
    heap_sort(arr3)