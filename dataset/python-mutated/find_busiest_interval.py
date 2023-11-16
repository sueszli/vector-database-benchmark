"""
Find the Busiest Interval

Given a list of arriving time and leaving time for each celebrity.
Celebrity I, arrives at arriving[I] time and leaves at leaving[I] time.
Output is the time interval that you want to go the party when the maximum number of celebrities are still there.

Input: arriving=[30, 0, 60], leaving=[75, 50, 150]
Output: (30, 50) or (60, 75)

=========================================
Just sort the lists, don't care about pairs ordering.
And use a counter, when arriving counter++, when leaving counter--.
    Time Complexity:    O(N LogN)
    Space Complexity:   O(1)
"""

def bussiest_interval(arriving, leaving):
    if False:
        while True:
            i = 10
    arriving.sort()
    leaving.sort()
    n = len(arriving)
    (i, j) = (0, 0)
    (start, end) = (0, 0)
    overlapping = 0
    max_overlapping = 0
    while i < n:
        if arriving[i] <= leaving[j]:
            overlapping += 1
            if max_overlapping <= overlapping:
                max_overlapping = overlapping
                start = arriving[i]
            i += 1
        else:
            if max_overlapping == overlapping:
                end = leaving[j]
            overlapping -= 1
            j += 1
    if max_overlapping == overlapping:
        end = leaving[j]
    return (start, end)
print(bussiest_interval([30, 0, 60], [75, 50, 150]))
print(bussiest_interval([1, 2, 10, 5, 5], [4, 5, 12, 9, 12]))