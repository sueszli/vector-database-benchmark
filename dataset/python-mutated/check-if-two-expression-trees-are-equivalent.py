import collections
import functools

class Node(object):

    def __init__(self, val=' ', left=None, right=None):
        if False:
            while True:
                i = 10
        pass

class Solution(object):

    def checkEquivalence(self, root1, root2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root1: Node\n        :type root2: Node\n        :rtype: bool\n        '

        def add_counter(counter, prev, d, val):
            if False:
                i = 10
                return i + 15
            if val.isalpha():
                counter[ord(val) - ord('a')] += d if prev[0] == '+' else -d
            prev[0] = val

        def morris_inorder_traversal(root, cb):
            if False:
                while True:
                    i = 10
            curr = root
            while curr:
                if curr.left is None:
                    cb(curr.val)
                    curr = curr.right
                else:
                    node = curr.left
                    while node.right and node.right != curr:
                        node = node.right
                    if node.right is None:
                        node.right = curr
                        curr = curr.left
                    else:
                        cb(curr.val)
                        node.right = None
                        curr = curr.right
        counter = collections.defaultdict(int)
        morris_inorder_traversal(root1, functools.partial(add_counter, counter, ['+'], 1))
        morris_inorder_traversal(root2, functools.partial(add_counter, counter, ['+'], -1))
        return all((v == 0 for v in counter.itervalues()))
import collections
import functools

class Solution2(object):

    def checkEquivalence(self, root1, root2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root1: Node\n        :type root2: Node\n        :rtype: bool\n        '

        def add_counter(counter, prev, d, val):
            if False:
                print('Hello World!')
            if val.isalpha():
                counter[ord(val) - ord('a')] += d if prev[0] == '+' else -d
            prev[0] = val

        def inorder_traversal(root, cb):
            if False:
                for i in range(10):
                    print('nop')

            def traverseLeft(node, stk):
                if False:
                    while True:
                        i = 10
                while node:
                    stk.append(node)
                    node = node.left
            stk = []
            traverseLeft(root, stk)
            while stk:
                curr = stk.pop()
                cb(curr.val)
                traverseLeft(curr.right, stk)
        counter = collections.defaultdict(int)
        inorder_traversal(root1, functools.partial(add_counter, counter, ['+'], 1))
        inorder_traversal(root2, functools.partial(add_counter, counter, ['+'], -1))
        return all((v == 0 for v in counter.itervalues()))