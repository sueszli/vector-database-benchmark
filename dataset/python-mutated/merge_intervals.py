"""
Merge Intervals

You are given an array of intervals.
Each interval is defined as: (start, end). e.g. (2, 5)
It represents all the integer numbers in the interval, including start and end. (in the example 2, 3, 4 and 5).
Given the array of intervals find the smallest set of unique intervals that contain the same integer numbers, without overlapping.


Input: [(1, 5), (2, 6)]
Output: [(1, 6)]

Input: [(2, 4), (5, 5), (6, 8)]
Output: [(2, 8)]

Input: [(1, 4), (6, 9), (8, 10)]
Output: [(1, 4), (6, 10)]

=========================================
Sort the intervals (using the start), accessing order. After that just iterate the intervals
and check if the current interval belongs to the last created interval.
    Time Complexity:    O(N LogN)
    Space Complexity:   O(N)    , for the result
"""

def merge_intervals(intervals):
    if False:
        for i in range(10):
            print('nop')
    n = len(intervals)
    if n == 0:
        return []
    intervals.sort(key=lambda interval: interval[0])
    mergedIntervals = []
    mergedIntervals.append(intervals[0])
    for i in range(1, n):
        if intervals[i][0] <= mergedIntervals[-1][1] + 1:
            mergedIntervals[-1] = (mergedIntervals[-1][0], max(mergedIntervals[-1][1], intervals[i][1]))
        else:
            mergedIntervals.append(intervals[i])
    return mergedIntervals
print(merge_intervals([(1, 5), (2, 6)]))
print(merge_intervals([(2, 4), (5, 5), (6, 8)]))
print(merge_intervals([(1, 4), (6, 9), (8, 10)]))