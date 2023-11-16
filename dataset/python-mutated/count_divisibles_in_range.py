"""
Count Divisibles in Range

Let us take a breather and tackle a problem so simple that its solution needs only a couple of
conditions, but not even any loops, let alone anything even more fancy. The difficulty is coming up
with the conditions that cover all possible cases of this problem exactly right, including all of the
potentially tricksy edge and corner cases, and not be off-by-one. Given three integers start, end
and n so that start <= end, count and return how many integers between start and end,
inclusive, are divisible by n.

Input: 7, 28, 4
Output: 6

=========================================
Find the close divisible to start (the smallest divisible in the range), calculate the difference between
that number and the end of the range, and in the end divide the difference by N.
    Time Complexity:    O(1)
    Space Complexity:   O(1)
"""

def count_divisibles_in_range(start, end, n):
    if False:
        print('Hello World!')
    start += (n - start % n) % n
    if start > end:
        return 0
    return 1 + (end - start) // n
print(count_divisibles_in_range(7, 28, 4))
print(count_divisibles_in_range(-77, 19, 10))
print(count_divisibles_in_range(-19, -13, 10))
print(count_divisibles_in_range(1, 10 ** 12 - 1, 5))
print(count_divisibles_in_range(0, 10 ** 12 - 1, 5))
print(count_divisibles_in_range(0, 10 ** 12, 5))