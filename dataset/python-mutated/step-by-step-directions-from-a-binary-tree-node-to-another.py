class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            while True:
                i = 10
        self.val = val
        self.left = left
        self.right = right

class Solution(object):

    def getDirections(self, root, startValue, destValue):
        if False:
            return 10
        '\n        :type root: Optional[TreeNode]\n        :type startValue: int\n        :type destValue: int\n        :rtype: str\n        '

        def iter_dfs(root, val):
            if False:
                print('Hello World!')
            path = []
            stk = [(1, (root,))]
            while stk:
                (step, args) = stk.pop()
                if step == 1:
                    node = args[0]
                    if node.val == val:
                        path.reverse()
                        return path
                    for (i, child) in enumerate((node.left, node.right)):
                        if not child:
                            continue
                        stk.append((3, None))
                        stk.append((1, (child,)))
                        stk.append((2, ('LR'[i],)))
                elif step == 2:
                    path.append(args[0])
                elif step == 3:
                    path.pop()
            return []
        src = iter_dfs(root, startValue)
        dst = iter_dfs(root, destValue)
        while len(src) and len(dst) and (src[-1] == dst[-1]):
            src.pop()
            dst.pop()
        dst.reverse()
        return ''.join(['U'] * len(src) + dst)

class Solution2(object):

    def getDirections(self, root, startValue, destValue):
        if False:
            return 10
        '\n        :type root: Optional[TreeNode]\n        :type startValue: int\n        :type destValue: int\n        :rtype: str\n        '

        def dfs(node, val, path):
            if False:
                return 10
            if node.val == val:
                return True
            if node.left and dfs(node.left, val, path):
                path.append('L')
            elif node.right and dfs(node.right, val, path):
                path.append('R')
            return path
        (src, dst) = ([], [])
        dfs(root, startValue, src)
        dfs(root, destValue, dst)
        while len(src) and len(dst) and (src[-1] == dst[-1]):
            src.pop()
            dst.pop()
        dst.reverse()
        return ''.join(['U'] * len(src) + dst)