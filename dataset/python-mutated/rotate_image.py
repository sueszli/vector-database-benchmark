"""
You are given an n x n 2D mat representing an image.

Rotate the image by 90 degrees (clockwise).

Follow up:
Could you do this in-place?
"""

def rotate(mat):
    if False:
        i = 10
        return i + 15
    if not mat:
        return mat
    mat.reverse()
    for i in range(len(mat)):
        for j in range(i):
            (mat[i][j], mat[j][i]) = (mat[j][i], mat[i][j])
    return mat
if __name__ == '__main__':
    mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(mat)
    rotate(mat)
    print(mat)