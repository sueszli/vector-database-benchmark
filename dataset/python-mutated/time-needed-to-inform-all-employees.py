import collections

class Solution(object):

    def numOfMinutes(self, n, headID, manager, informTime):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type headID: int\n        :type manager: List[int]\n        :type informTime: List[int]\n        :rtype: int\n        '
        children = collections.defaultdict(list)
        for (child, parent) in enumerate(manager):
            if parent != -1:
                children[parent].append(child)
        result = 0
        stk = [(headID, 0)]
        while stk:
            (node, curr) = stk.pop()
            curr += informTime[node]
            result = max(result, curr)
            if node not in children:
                continue
            for c in children[node]:
                stk.append((c, curr))
        return result

class Solution2(object):

    def numOfMinutes(self, n, headID, manager, informTime):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type headID: int\n        :type manager: List[int]\n        :type informTime: List[int]\n        :rtype: int\n        '

        def dfs(informTime, children, node):
            if False:
                i = 10
                return i + 15
            return (max((dfs(informTime, children, c) for c in children[node])) if node in children else 0) + informTime[node]
        children = collections.defaultdict(list)
        for (child, parent) in enumerate(manager):
            if parent != -1:
                children[parent].append(child)
        return dfs(informTime, children, headID)