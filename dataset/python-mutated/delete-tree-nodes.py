import collections

class Solution(object):

    def deleteTreeNodes(self, nodes, parent, value):
        if False:
            return 10
        '\n        :type nodes: int\n        :type parent: List[int]\n        :type value: List[int]\n        :rtype: int\n        '

        def dfs(value, children, x):
            if False:
                for i in range(10):
                    print('nop')
            (total, count) = (value[x], 1)
            for y in children[x]:
                (t, c) = dfs(value, children, y)
                total += t
                count += c if t else 0
            return (total, count if total else 0)
        children = collections.defaultdict(list)
        for (i, p) in enumerate(parent):
            if i:
                children[p].append(i)
        return dfs(value, children, 0)[1]

class Solution2(object):

    def deleteTreeNodes(self, nodes, parent, value):
        if False:
            return 10
        '\n        :type nodes: int\n        :type parent: List[int]\n        :type value: List[int]\n        :rtype: int\n        '
        result = [1] * nodes
        for i in reversed(xrange(1, nodes)):
            value[parent[i]] += value[i]
            result[parent[i]] += result[i] if value[i] else 0
        return result[0]