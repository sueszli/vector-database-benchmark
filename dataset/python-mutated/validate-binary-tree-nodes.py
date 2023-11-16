class Solution(object):

    def validateBinaryTreeNodes(self, n, leftChild, rightChild):
        if False:
            return 10
        '\n        :type n: int\n        :type leftChild: List[int]\n        :type rightChild: List[int]\n        :rtype: bool\n        '
        roots = set(range(n)) - set(leftChild) - set(rightChild)
        if len(roots) != 1:
            return False
        (root,) = roots
        stk = [root]
        lookup = set([root])
        while stk:
            node = stk.pop()
            for c in (leftChild[node], rightChild[node]):
                if c < 0:
                    continue
                if c in lookup:
                    return False
                lookup.add(c)
                stk.append(c)
        return len(lookup) == n