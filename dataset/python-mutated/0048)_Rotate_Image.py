class Solution:

    def rotate(self, matrix: List[List[int]]) -> None:
        if False:
            print('Hello World!')
        matrix[:] = zip(*matrix[::-1])