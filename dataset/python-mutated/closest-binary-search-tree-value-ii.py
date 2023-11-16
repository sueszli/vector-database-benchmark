class Solution(object):

    def closestKValues(self, root, target, k):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :type target: float\n        :type k: int\n        :rtype: List[int]\n        '

        def nextNode(stack, child1, child2):
            if False:
                while True:
                    i = 10
            if stack:
                if child2(stack):
                    stack.append(child2(stack))
                    while child1(stack):
                        stack.append(child1(stack))
                else:
                    child = stack.pop()
                    while stack and child is child2(stack):
                        child = stack.pop()
        backward = lambda stack: stack[-1].left
        forward = lambda stack: stack[-1].right
        stack = []
        while root:
            stack.append(root)
            root = root.left if target < root.val else root.right
        dist = lambda node: abs(node.val - target)
        forward_stack = stack[:stack.index(min(stack, key=dist)) + 1]
        backward_stack = list(forward_stack)
        nextNode(backward_stack, backward, forward)
        result = []
        for _ in xrange(k):
            if forward_stack and (not backward_stack or dist(forward_stack[-1]) < dist(backward_stack[-1])):
                result.append(forward_stack[-1].val)
                nextNode(forward_stack, forward, backward)
            elif backward_stack and (not forward_stack or dist(backward_stack[-1]) <= dist(forward_stack[-1])):
                result.append(backward_stack[-1].val)
                nextNode(backward_stack, backward, forward)
        return result

class Solution2(object):

    def closestKValues(self, root, target, k):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :type target: float\n        :type k: int\n        :rtype: List[int]\n        '

        class BSTIterator:

            def __init__(self, stack, child1, child2):
                if False:
                    while True:
                        i = 10
                self.stack = list(stack)
                self.cur = self.stack.pop()
                self.child1 = child1
                self.child2 = child2

            def next(self):
                if False:
                    return 10
                node = None
                if self.cur and self.child1(self.cur):
                    self.stack.append(self.cur)
                    node = self.child1(self.cur)
                    while self.child2(node):
                        self.stack.append(node)
                        node = self.child2(node)
                elif self.stack:
                    prev = self.cur
                    node = self.stack.pop()
                    while node:
                        if self.child2(node) is prev:
                            break
                        else:
                            prev = node
                            node = self.stack.pop() if self.stack else None
                self.cur = node
                return node
        stack = []
        while root:
            stack.append(root)
            root = root.left if target < root.val else root.right
        dist = lambda node: abs(node.val - target) if node else float('inf')
        stack = stack[:stack.index(min(stack, key=dist)) + 1]
        backward = lambda node: node.left
        forward = lambda node: node.right
        (smaller_it, larger_it) = (BSTIterator(stack, backward, forward), BSTIterator(stack, forward, backward))
        (smaller_node, larger_node) = (smaller_it.next(), larger_it.next())
        result = [stack[-1].val]
        for _ in xrange(k - 1):
            if dist(smaller_node) < dist(larger_node):
                result.append(smaller_node.val)
                smaller_node = smaller_it.next()
            else:
                result.append(larger_node.val)
                larger_node = larger_it.next()
        return result