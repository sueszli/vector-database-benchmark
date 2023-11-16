"""
Given a list of sorted characters letters containing only lowercase letters,
and given a target letter target, find the smallest element in the list that
is larger than the given target.

Letters also wrap around. For example, if the target is target = 'z' and
letters = ['a', 'b'], the answer is 'a'.

Input:
letters = ["c", "f", "j"]
target = "a"
Output: "c"

Input:
letters = ["c", "f", "j"]
target = "c"
Output: "f"

Input:
letters = ["c", "f", "j"]
target = "d"
Output: "f"

Reference: https://leetcode.com/problems/find-smallest-letter-greater-than-target/description/
"""
import bisect

def next_greatest_letter(letters, target):
    if False:
        i = 10
        return i + 15
    '\n    Using bisect libarary\n    '
    index = bisect.bisect(letters, target)
    return letters[index % len(letters)]

def next_greatest_letter_v1(letters, target):
    if False:
        while True:
            i = 10
    '\n    Using binary search: complexity O(logN)\n    '
    if letters[0] > target:
        return letters[0]
    if letters[len(letters) - 1] <= target:
        return letters[0]
    (left, right) = (0, len(letters) - 1)
    while left <= right:
        mid = left + (right - left) // 2
        if letters[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return letters[left]

def next_greatest_letter_v2(letters, target):
    if False:
        print('Hello World!')
    '\n    Brute force: complexity O(N)\n    '
    for index in letters:
        if index > target:
            return index
    return letters[0]