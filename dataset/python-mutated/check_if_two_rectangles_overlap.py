"""
Check if Two Rectangles Overlap

Given two rectangles, find if the given two rectangles overlap or not.
Note that a rectangle can be represented by two coordinates, top left and bottom right.
So mainly we are given following four coordinates (min X and Y and max X and Y).
    - l1: Bottom Left coordinate of first rectangle. (mins)
    - r1: Top Right coordinate of first rectangle. (maxs)
    - l2: Bottom Left coordinate of second rectangle. (mins)
    - r2: Top Right coordinate of second rectangle. (maxs)
It may be assumed that the rectangles are PARALLEL to the coordinate axis.

Input: (0, 0), (3, 2), (1, 1), (5, 4)
Output: True

=========================================
First check if rectangles are overlapping on X axis and
after that if they are overlapping on Y axis.
    Time Complexity:    O(1)
    Space Complexity:   O(1)
"""

def check_if_two_rectangles_overlap(l1, r1, l2, r2):
    if False:
        i = 10
        return i + 15
    if r1[0] < l2[0] or l1[0] > r2[0]:
        return False
    if r1[1] < l2[1] or l1[1] > r2[1]:
        return False
    return True
print(check_if_two_rectangles_overlap((0, 0), (3, 2), (1, 1), (5, 4)))
print(check_if_two_rectangles_overlap((0, 0), (3, 2), (3, 2), (5, 4)))
print(check_if_two_rectangles_overlap((0, 0), (3, 2), (1, -1), (5, 4)))
print(check_if_two_rectangles_overlap((0, 0), (3, 2), (2, 3), (5, 4)))