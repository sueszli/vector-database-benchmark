class Solution(object):

    def kthSmallest(self, root, k):
        if False:
            i = 10
            return i + 15
        (s, cur, rank) = ([], root, 0)
        while s or cur:
            if cur:
                s.append(cur)
                cur = cur.left
            else:
                cur = s.pop()
                rank += 1
                if rank == k:
                    return cur.val
                cur = cur.right
        return float('-inf')
from itertools import islice

class Solution2(object):

    def kthSmallest(self, root, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :type k: int\n        :rtype: int\n        '

        def gen_inorder(root):
            if False:
                i = 10
                return i + 15
            if root:
                for n in gen_inorder(root.left):
                    yield n
                yield root.val
                for n in gen_inorder(root.right):
                    yield n
        return next(islice(gen_inorder(root), k - 1, k))