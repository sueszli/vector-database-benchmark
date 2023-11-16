import collections

class Solution(object):

    def isEscapePossible(self, blocked, source, target):
        if False:
            while True:
                i = 10
        '\n        :type blocked: List[List[int]]\n        :type source: List[int]\n        :type target: List[int]\n        :rtype: bool\n        '
        (R, C) = (10 ** 6, 10 ** 6)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        def bfs(blocks, source, target):
            if False:
                return 10
            max_area_surrounded_by_blocks = len(blocks) * (len(blocks) - 1) // 2
            lookup = set([source])
            if len(lookup) > max_area_surrounded_by_blocks:
                return True
            q = collections.deque([source])
            while q:
                source = q.popleft()
                if source == target:
                    return True
                for direction in directions:
                    (nr, nc) = (source[0] + direction[0], source[1] + direction[1])
                    if not (0 <= nr < R and 0 <= nc < C and ((nr, nc) not in lookup) and ((nr, nc) not in blocks)):
                        continue
                    lookup.add((nr, nc))
                    if len(lookup) > max_area_surrounded_by_blocks:
                        return True
                    q.append((nr, nc))
            return False
        return bfs(set(map(tuple, blocked)), tuple(source), tuple(target)) and bfs(set(map(tuple, blocked)), tuple(target), tuple(source))