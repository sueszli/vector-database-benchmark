"""
Perfect Rectangle

Given N axis-aligned rectangles where N > 0, determine if they all together form an exact cover of a rectangular region.
Each rectangle is represented as a bottom-left point and a top-right point. For example, a unit square is represented as [1,1,2,2].
(coordinate of bottom-left point is (1, 1) and top-right point is (2, 2)).

Input: [
        [1, 1, 3, 3],
        [3, 1, 4, 2],
        [3, 2, 4, 4],
        [1, 3, 2, 4],
        [2, 3, 3, 4]
    ]
Output: True
Output explanation: All 5 rectangles together form an exact cover of a rectangular region.

Input: [
        [1, 1, 2, 3],
        [1, 3, 2, 4],
        [3, 1, 4, 2],
        [3, 2, 4, 4]
    ]
Output: False
Output explanation: Because there is a gap between the two rectangular regions.

Input: [
        [1, 1, 3, 3],
        [3, 1, 4, 2],
        [1, 3, 2, 4],
        [3, 2, 4, 4]
    ]
Output: False
Output explanation: Because there is a gap in the top center.

Input: [
        [1, 1, 3, 3],
        [3, 1, 4, 2],
        [1, 3, 2, 4],
        [2, 2, 4, 4]
    ]
Output: False
Output explanation: Because two of the rectangles overlap with each other.

=========================================
Check if 4 unique points exist. If 4 unique points exist, then
check if the sum of all rectangles is equal to the final rectangle.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""
import math

def is_perfect_rectangle(rectangles):
    if False:
        print('Hello World!')
    areas_sum = 0
    all_points = set()
    for rect in rectangles:
        areas_sum += (rect[2] - rect[0]) * (rect[3] - rect[1])
        rect_points = [(rect[0], rect[1]), (rect[0], rect[3]), (rect[2], rect[3]), (rect[2], rect[1])]
        for point in rect_points:
            if point in all_points:
                all_points.remove(point)
            else:
                all_points.add(point)
    if len(all_points) != 4:
        return False
    bounding_rectangle = [math.inf, math.inf, -math.inf, -math.inf]
    for point in all_points:
        bounding_rectangle = [min(bounding_rectangle[0], point[0]), min(bounding_rectangle[1], point[1]), max(bounding_rectangle[2], point[0]), max(bounding_rectangle[3], point[1])]
    bounding_rectangle_area = (bounding_rectangle[2] - bounding_rectangle[0]) * (bounding_rectangle[3] - bounding_rectangle[1])
    return areas_sum == bounding_rectangle_area
rectangles = [[1, 1, 3, 3], [3, 1, 4, 2], [3, 2, 4, 4], [1, 3, 2, 4], [2, 3, 3, 4]]
print(is_perfect_rectangle(rectangles))
rectangles = [[1, 1, 2, 3], [1, 3, 2, 4], [3, 1, 4, 2], [3, 2, 4, 4]]
print(is_perfect_rectangle(rectangles))
rectangles = [[1, 1, 3, 3], [3, 1, 4, 2], [1, 3, 2, 4], [3, 2, 4, 4]]
print(is_perfect_rectangle(rectangles))
rectangles = [[1, 1, 3, 3], [3, 1, 4, 2], [1, 3, 2, 4], [2, 2, 4, 4]]
print(is_perfect_rectangle(rectangles))