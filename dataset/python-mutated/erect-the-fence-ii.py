import random

class Solution(object):

    def outerTrees(self, trees):
        if False:
            while True:
                i = 10
        '\n        :type trees: List[List[int]]\n        :rtype: List[float]\n        '

        def dist(a, b):
            if False:
                while True:
                    i = 10
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

        def inside(c, p):
            if False:
                while True:
                    i = 10
            return dist(c[0], p) < c[1] + EPS

        def circle_center(bx, by, cx, cy):
            if False:
                i = 10
                return i + 15
            B = bx * bx + by * by
            C = cx * cx + cy * cy
            D = bx * cy - by * cx
            return [float(cy * B - by * C) / (2 * D), float(bx * C - cx * B) / (2 * D)]

        def circle_from_2_points(A, B):
            if False:
                for i in range(10):
                    print('nop')
            C = [(A[0] + B[0]) / 2.0, (A[1] + B[1]) / 2.0]
            return [C, dist(A, B) / 2.0]

        def circle_from_3_points(A, B, C):
            if False:
                i = 10
                return i + 15
            I = circle_center(B[0] - A[0], B[1] - A[1], C[0] - A[0], C[1] - A[1])
            I[0] += A[0]
            I[1] += A[1]
            return [I, dist(I, A)]

        def trivial(boundaries):
            if False:
                i = 10
                return i + 15
            if not boundaries:
                return None
            if len(boundaries) == 1:
                return [boundaries[0], 0.0]
            if len(boundaries) == 2:
                return circle_from_2_points(boundaries[0], boundaries[1])
            return circle_from_3_points(boundaries[0], boundaries[1], boundaries[2])

        def Welzl(points, boundaries, curr):
            if False:
                print('Hello World!')
            if curr == len(points) or len(boundaries) == 3:
                return trivial(boundaries)
            result = Welzl(points, boundaries, curr + 1)
            if result is not None and inside(result, points[curr]):
                return result
            boundaries.append(points[curr])
            result = Welzl(points, boundaries, curr + 1)
            boundaries.pop()
            return result
        EPS = 1e-05
        random.seed(0)
        random.shuffle(trees)
        result = Welzl(trees, [], 0)
        return (result[0][0], result[0][1], result[1])