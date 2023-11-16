"""
Top K Frequent Elements

Given a non-empty array of integers, return the k most frequent elements.
The order of the result isn't important.

Input: [1, 1, 1, 2, 2, 3], 2
Output: [1, 2]

Input: [1], 1
Output: [1]

=========================================
Using Min Priority Queue, in each step add an element with its frequency and remove the element with the smallest frequency
if there are more than K elements inside the Priority Queue. This solution isn't much faster than sorting the frequencies.
    Time Complexity:    O(U LogK)   , U in this case is the number of unique elements (but all elements from the array could be unique, so because of that U can be equal to N)
    Space Complexity:   O(N)
Using pivoting, this solution is based on the quick sort algorithm (divide and conquer).
This algorithm is called: QucikSelect - The quicksort pivoting logic but for searching kth smallest (not sorting the whole array) - O(n) complexity (n + n/2 + n/4 + ... + 1 = 2n)
https://en.wikipedia.org/wiki/Quickselect
Same solution as kth_smallest.py.
    Time Complexity:    O(U)
    Space Complexity:   O(N)
"""
import heapq

class PQElement:

    def __init__(self, el):
        if False:
            while True:
                i = 10
        (self.frequency, self.val) = el

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return self.frequency < other.frequency

class PriorityQueue:

    def __init__(self):
        if False:
            return 10
        self.data = []

    def push(self, el):
        if False:
            print('Hello World!')
        heapq.heappush(self.data, PQElement(el))

    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        return heapq.heappop(self.data)

    def count(self):
        if False:
            i = 10
            return i + 15
        return len(self.data)

def top_k_frequent_1(nums, k):
    if False:
        i = 10
        return i + 15
    frequency = {}
    for num in nums:
        if num in frequency:
            frequency[num] += 1
        else:
            frequency[num] = 1
    arr = [(frequency[el], el) for el in frequency]
    n = len(arr)
    if k > n:
        return [el[1] for el in arr]
    if k < 1:
        return []
    heap = PriorityQueue()
    for el in arr:
        heap.push(el)
        if heap.count() > k:
            heap.pop()
    return [el.val for el in heap.data]

def top_k_frequent_2(nums, k):
    if False:
        for i in range(10):
            print('nop')
    frequency = {}
    for num in nums:
        if num in frequency:
            frequency[num] += 1
        else:
            frequency[num] = 1
    arr = [(frequency[el], el) for el in frequency]
    n = len(arr)
    if k > n:
        return [el[1] for el in arr]
    if k < 1:
        return []
    k -= 1
    left = 0
    right = n - 1
    while True:
        pivot = pivoting(arr, left, right)
        if pivot > k:
            right = pivot - 1
        elif pivot < k:
            left = pivot + 1
        else:
            return [el[1] for el in arr[:k + 1]]
    return None

def pivoting(arr, left, right):
    if False:
        return 10
    pivot = right
    new_pivot = left
    for j in range(left, right):
        if arr[j][0] > arr[pivot][0]:
            swap(arr, new_pivot, j)
            new_pivot += 1
    swap(arr, new_pivot, pivot)
    return new_pivot

def swap(arr, i, j):
    if False:
        for i in range(10):
            print('nop')
    (arr[i], arr[j]) = (arr[j], arr[i])
print(top_k_frequent_1([1, 1, 1, 2, 2, 3], 2))
print(top_k_frequent_2([1, 1, 1, 2, 2, 3], 2))
print(top_k_frequent_1([1], 1))
print(top_k_frequent_2([1], 1))