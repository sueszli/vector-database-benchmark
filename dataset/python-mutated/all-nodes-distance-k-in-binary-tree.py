import collections

class Solution(object):

    def distanceK(self, root, target, K):
        if False:
            return 10
        '\n        :type root: TreeNode\n        :type target: TreeNode\n        :type K: int\n        :rtype: List[int]\n        '

        def dfs(parent, child, neighbors):
            if False:
                while True:
                    i = 10
            if not child:
                return
            if parent:
                neighbors[parent.val].append(child.val)
                neighbors[child.val].append(parent.val)
            dfs(child, child.left, neighbors)
            dfs(child, child.right, neighbors)
        neighbors = collections.defaultdict(list)
        dfs(None, root, neighbors)
        bfs = [target.val]
        lookup = set(bfs)
        for _ in xrange(K):
            bfs = [nei for node in bfs for nei in neighbors[node] if nei not in lookup]
            lookup |= set(bfs)
        return bfs