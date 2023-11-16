import itertools

class TreeNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def getTargetCopy(self, original, cloned, target):
        if False:
            print('Hello World!')
        '\n        :type original: TreeNode\n        :type cloned: TreeNode\n        :type target: TreeNode\n        :rtype: TreeNode\n        '

        def preorder_gen(node):
            if False:
                print('Hello World!')
            stk = [node]
            while stk:
                node = stk.pop()
                if not node:
                    continue
                yield node
                stk.append(node.right)
                stk.append(node.left)
        for (node1, node2) in itertools.izip(preorder_gen(original), preorder_gen(cloned)):
            if node1 == target:
                return node2