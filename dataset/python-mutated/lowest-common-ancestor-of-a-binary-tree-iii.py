class Node:

    def __init__(self, val):
        if False:
            i = 10
            return i + 15
        pass

class Solution(object):

    def lowestCommonAncestor(self, p, q):
        if False:
            print('Hello World!')
        '\n        :type node: Node\n        :rtype: Node\n        '
        (a, b) = (p, q)
        while a != b:
            a = a.parent if a else q
            b = b.parent if b else p
        return a

class Solution2(object):

    def lowestCommonAncestor(self, p, q):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type node: Node\n        :rtype: Node\n        '

        def depth(node):
            if False:
                for i in range(10):
                    print('nop')
            d = 0
            while node:
                node = node.parent
                d += 1
            return d
        (p_d, q_d) = (depth(p), depth(q))
        while p_d > q_d:
            p = p.parent
            p_d -= 1
        while p_d < q_d:
            q = q.parent
            q_d -= 1
        while p != q:
            p = p.parent
            q = q.parent
        return p