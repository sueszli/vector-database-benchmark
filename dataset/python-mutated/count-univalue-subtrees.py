class Solution(object):

    def countUnivalSubtrees(self, root):
        if False:
            print('Hello World!')
        [is_uni, count] = self.isUnivalSubtrees(root, 0)
        return count

    def isUnivalSubtrees(self, root, count):
        if False:
            i = 10
            return i + 15
        if not root:
            return [True, count]
        [left, count] = self.isUnivalSubtrees(root.left, count)
        [right, count] = self.isUnivalSubtrees(root.right, count)
        if self.isSame(root, root.left, left) and self.isSame(root, root.right, right):
            count += 1
            return [True, count]
        return [False, count]

    def isSame(self, root, child, is_uni):
        if False:
            print('Hello World!')
        return not child or (is_uni and root.val == child.val)