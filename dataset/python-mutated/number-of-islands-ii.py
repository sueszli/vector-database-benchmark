class Solution(object):

    def numIslands2(self, m, n, positions):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type m: int\n        :type n: int\n        :type positions: List[List[int]]\n        :rtype: List[int]\n        '

        def node_id(node, n):
            if False:
                return 10
            return node[0] * n + node[1]

        def find_set(x):
            if False:
                i = 10
                return i + 15
            if set[x] != x:
                set[x] = find_set(set[x])
            return set[x]

        def union_set(x, y):
            if False:
                for i in range(10):
                    print('nop')
            (x_root, y_root) = (find_set(x), find_set(y))
            set[min(x_root, y_root)] = max(x_root, y_root)
        numbers = []
        number = 0
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        set = {}
        for position in positions:
            node = (position[0], position[1])
            set[node_id(node, n)] = node_id(node, n)
            number += 1
            for d in directions:
                neighbor = (position[0] + d[0], position[1] + d[1])
                if 0 <= neighbor[0] < m and 0 <= neighbor[1] < n and (node_id(neighbor, n) in set):
                    if find_set(node_id(node, n)) != find_set(node_id(neighbor, n)):
                        union_set(node_id(node, n), node_id(neighbor, n))
                        number -= 1
            numbers.append(number)
        return numbers