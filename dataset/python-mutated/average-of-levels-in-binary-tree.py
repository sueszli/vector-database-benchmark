class Solution(object):

    def averageOfLevels(self, root):
        if False:
            return 10
        '\n        :type root: TreeNode\n        :rtype: List[float]\n        '
        result = []
        q = [root]
        while q:
            (total, count) = (0, 0)
            next_q = []
            for n in q:
                total += n.val
                count += 1
                if n.left:
                    next_q.append(n.left)
                if n.right:
                    next_q.append(n.right)
            q = next_q
            result.append(float(total) / count)
        return result