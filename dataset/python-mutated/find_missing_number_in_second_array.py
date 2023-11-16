"""
Find missing number in second array

Given 2 arrays, first array with N elements and second array with N-1 elements.
All elements from the first array exist in the second array, except one. Find the missing number.

Sample input:   [1, 2, 3, 4, 5], [1, 2, 3, 4]
Sample output:  5

Sample input:   [2131, 2122221, 64565, 33333333, 994188129, 865342234],
                [994188129, 2122221, 865342234, 2131, 64565]
Sample output:  33333333

=========================================
The simplest solution is to substract the sum of the second array from the sum of the first array:
missing_number = sum(arr1) - sum(arr2)
But what if we have milions of elements and all elements are with 8-9 digits values?
In this case we'll need to use modulo operation. Make two sums, the first one from MODULO of each element
and the second one from the DIVIDE of each element. In the end use these 2 sums to compute the missing number.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
The second solution is XOR soulution, make XOR to each element from the both arrays (same as find_unpaired.py).
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def find_missing_number(arr1, arr2):
    if False:
        i = 10
        return i + 15
    n = len(arr2)
    mod = 10000
    sum_diff = 0
    mod_diff = 0
    i = 0
    while i < n:
        sum_diff += arr1[i] % mod - arr2[i] % mod
        mod_diff += arr1[i] // mod - arr2[i] // mod
        i += 1
    sum_diff += arr1[n] % mod
    mod_diff += arr1[n] // mod
    return mod * mod_diff + sum_diff

def find_missing_number_2(arr1, arr2):
    if False:
        i = 10
        return i + 15
    n = len(arr2)
    missing = 0
    i = 0
    while i < n:
        missing ^= arr1[i] ^ arr2[i]
        i += 1
    missing ^= arr1[n]
    return missing
arr1 = [2131, 2122221, 64565, 33333333, 994188129, 865342234]
arr2 = [994188129, 2122221, 865342234, 2131, 64565]
print(find_missing_number(arr1, arr2))
print(find_missing_number_2(arr1, arr2))
arr1 = [1, 2, 3, 4, 5]
arr2 = [1, 2, 3, 4]
print(find_missing_number(arr1, arr2))
print(find_missing_number_2(arr1, arr2))