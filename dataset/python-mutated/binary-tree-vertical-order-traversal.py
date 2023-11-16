import collections

class Solution(object):

    def verticalOrder(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: TreeNode\n        :rtype: List[List[int]]\n        '
        cols = collections.defaultdict(list)
        queue = [(root, 0)]
        for (node, i) in queue:
            if node:
                cols[i].append(node.val)
                queue += ((node.left, i - 1), (node.right, i + 1))
        return [cols[i] for i in xrange(min(cols.keys()), max(cols.keys()) + 1)] if cols else []