class ListNode(object):

    def __init__(self, val=0, next=None):
        if False:
            i = 10
            return i + 15
        pass

class Solution(object):

    def spiralMatrix(self, m, n, head):
        if False:
            return 10
        '\n        :type m: int\n        :type n: int\n        :type head: Optional[ListNode]\n        :rtype: List[List[int]]\n        '
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        result = [[-1] * n for _ in xrange(m)]
        i = j = d = 0
        while head:
            result[i][j] = head.val
            if not (0 <= i + directions[d][0] < m and 0 <= j + directions[d][1] < n and (result[i + directions[d][0]][j + directions[d][1]] == -1)):
                d = (d + 1) % 4
            (i, j) = (i + directions[d][0], j + directions[d][1])
            head = head.next
        return result