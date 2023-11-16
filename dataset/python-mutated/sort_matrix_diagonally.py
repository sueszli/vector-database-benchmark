"""
Given a m * n matrix mat of integers,
sort it diagonally in ascending order
from the top-left to the bottom-right
then return the sorted array.

mat = [
    [3,3,1,1],
    [2,2,1,2],
    [1,1,1,2]
]

Should return:
[
    [1,1,1,1],
    [1,2,2,2],
    [1,2,3,3]
]
"""
from heapq import heappush, heappop
from typing import List

def sort_diagonally(mat: List[List[int]]) -> List[List[int]]:
    if False:
        return 10
    if len(mat) == 1 or len(mat[0]) == 1:
        return mat
    for i in range(len(mat) + len(mat[0]) - 1):
        if i + 1 < len(mat):
            h = []
            row = len(mat) - (i + 1)
            col = 0
            while row < len(mat):
                heappush(h, mat[row][col])
                row += 1
                col += 1
            row = len(mat) - (i + 1)
            col = 0
            while h:
                ele = heappop(h)
                mat[row][col] = ele
                row += 1
                col += 1
        else:
            h = []
            row = 0
            col = i - (len(mat) - 1)
            while col < len(mat[0]) and row < len(mat):
                heappush(h, mat[row][col])
                row += 1
                col += 1
            row = 0
            col = i - (len(mat) - 1)
            while h:
                ele = heappop(h)
                mat[row][col] = ele
                row += 1
                col += 1
    return mat