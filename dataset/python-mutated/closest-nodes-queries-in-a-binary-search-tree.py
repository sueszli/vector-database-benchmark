import bisect

class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            print('Hello World!')
        pass

class Solution(object):

    def closestNodes(self, root, queries):
        if False:
            i = 10
            return i + 15
        '\n        :type root: Optional[TreeNode]\n        :type queries: List[int]\n        :rtype: List[List[int]]\n        '

        def iter_dfs():
            if False:
                for i in range(10):
                    print('nop')
            inorder = []
            stk = [(1, root)]
            while stk:
                (step, node) = stk.pop()
                if step == 1:
                    if not node:
                        continue
                    stk.append((1, node.right))
                    stk.append((2, node))
                    stk.append((1, node.left))
                elif step == 2:
                    inorder.append(node.val)
            return inorder
        inorder = iter_dfs()
        result = []
        for q in queries:
            i = bisect.bisect_left(inorder, q)
            if i == len(inorder):
                result.append([inorder[i - 1], -1])
            elif inorder[i] == q:
                result.append([inorder[i], inorder[i]])
            elif i - 1 >= 0:
                result.append([inorder[i - 1], inorder[i]])
            else:
                result.append([-1, inorder[i]])
        return result
import bisect

class Solution2(object):

    def closestNodes(self, root, queries):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: Optional[TreeNode]\n        :type queries: List[int]\n        :rtype: List[List[int]]\n        '

        def dfs(node):
            if False:
                print('Hello World!')
            if not node:
                return
            dfs(node.left)
            inorder.append(node.val)
            dfs(node.right)
        inorder = []
        dfs(root)
        result = []
        for q in queries:
            i = bisect.bisect_left(inorder, q)
            if i == len(inorder):
                result.append([inorder[i - 1], -1])
            elif inorder[i] == q:
                result.append([inorder[i], inorder[i]])
            elif i - 1 >= 0:
                result.append([inorder[i - 1], inorder[i]])
            else:
                result.append([-1, inorder[i]])
        return result