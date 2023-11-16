"""
Majority Element

Given an array of size n, find the majority element.
The majority element is the element that appears more than ⌊ n/2 ⌋ times.
You may assume that the array is non-empty and the majority element always exist in the array.

Input: [3, 2, 3]
Output: 3

Input: [2, 2, 1, 1, 1, 2, 2]
Output: 2

=========================================
Sort the array and the result will the middle element.
    Time Complexity:    O(N LogN)
    Space Complexity:   O(1)
Use dictionary (hash map) and count the occurrences.
The result will be the one with more than ⌊ n/2 ⌋ occurrences.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
Using Boyer-Moore voting algorithm. Choose a potential majority element and for each occurence add +1, but
if the current element isn't same substract -1.
When the counter is 0, the next element becomes the new potential majority element.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def majority_element_1(nums):
    if False:
        return 10
    nums.sort()
    return nums[len(nums) // 2]

def majority_element_2(nums):
    if False:
        print('Hello World!')
    counter = {}
    for num in nums:
        if num in counter:
            counter[num] += 1
        else:
            counter[num] = 1
    half = len(nums) // 2
    for num in counter:
        if counter[num] > half:
            return num

def majority_element_3(nums):
    if False:
        while True:
            i = 10
    majority = 0
    count = 0
    for num in nums:
        if count == 0:
            majority = num
        if num == majority:
            count += 1
        else:
            count -= 1
    return majority
arr = [3, 2, 3]
print(majority_element_1(arr))
print(majority_element_2(arr))
print(majority_element_3(arr))
arr = [2, 2, 1, 1, 1, 2, 2]
print(majority_element_1(arr))
print(majority_element_2(arr))
print(majority_element_3(arr))