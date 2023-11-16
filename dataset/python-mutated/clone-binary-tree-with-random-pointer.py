class Node(object):

    def __init__(self, val=0, left=None, right=None, random=None):
        if False:
            print('Hello World!')
        self.val = val
        self.left = left
        self.right = right
        self.random = random

class NodeCopy(object):

    def __init__(self, val=0, left=None, right=None, random=None):
        if False:
            while True:
                i = 10
        pass

class Solution(object):

    def copyRandomBinaryTree(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: Node\n        :rtype: NodeCopy\n        '

        def iter_dfs(node, callback):
            if False:
                print('Hello World!')
            result = None
            stk = [node]
            while stk:
                node = stk.pop()
                if not node:
                    continue
                (left_node, copy) = callback(node)
                if not result:
                    result = copy
                stk.append(node.right)
                stk.append(left_node)
            return result

        def merge(node):
            if False:
                i = 10
                return i + 15
            copy = NodeCopy(node.val)
            (node.left, copy.left) = (copy, node.left)
            return (copy.left, copy)

        def clone(node):
            if False:
                print('Hello World!')
            copy = node.left
            node.left.random = node.random.left if node.random else None
            node.left.right = node.right.left if node.right else None
            return (copy.left, copy)

        def split(node):
            if False:
                return 10
            copy = node.left
            (node.left, copy.left) = (copy.left, copy.left.left if copy.left else None)
            return (node.left, copy)
        iter_dfs(root, merge)
        iter_dfs(root, clone)
        return iter_dfs(root, split)

class Solution_Recu(object):

    def copyRandomBinaryTree(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: Node\n        :rtype: NodeCopy\n        '

        def dfs(node, callback):
            if False:
                i = 10
                return i + 15
            if not node:
                return None
            (left_node, copy) = callback(node)
            dfs(left_node, callback)
            dfs(node.right, callback)
            return copy

        def merge(node):
            if False:
                return 10
            copy = NodeCopy(node.val)
            (node.left, copy.left) = (copy, node.left)
            return (copy.left, copy)

        def clone(node):
            if False:
                while True:
                    i = 10
            copy = node.left
            node.left.random = node.random.left if node.random else None
            node.left.right = node.right.left if node.right else None
            return (copy.left, copy)

        def split(node):
            if False:
                print('Hello World!')
            copy = node.left
            (node.left, copy.left) = (copy.left, copy.left.left if copy.left else None)
            return (node.left, copy)
        dfs(root, merge)
        dfs(root, clone)
        return dfs(root, split)
import collections

class Solution2(object):

    def copyRandomBinaryTree(self, root):
        if False:
            return 10
        '\n        :type root: Node\n        :rtype: NodeCopy\n        '
        lookup = collections.defaultdict(lambda : NodeCopy())
        lookup[None] = None
        stk = [root]
        while stk:
            node = stk.pop()
            if not node:
                continue
            lookup[node].val = node.val
            lookup[node].left = lookup[node.left]
            lookup[node].right = lookup[node.right]
            lookup[node].random = lookup[node.random]
            stk.append(node.right)
            stk.append(node.left)
        return lookup[root]
import collections

class Solution2_Recu(object):

    def copyRandomBinaryTree(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: Node\n        :rtype: NodeCopy\n        '

        def dfs(node, lookup):
            if False:
                while True:
                    i = 10
            if not node:
                return
            lookup[node].val = node.val
            lookup[node].left = lookup[node.left]
            lookup[node].right = lookup[node.right]
            lookup[node].random = lookup[node.random]
            dfs(node.left, lookup)
            dfs(node.right, lookup)
        lookup = collections.defaultdict(lambda : NodeCopy())
        lookup[None] = None
        dfs(root, lookup)
        return lookup[root]