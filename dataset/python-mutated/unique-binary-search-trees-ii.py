class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

    def __repr__(self):
        if False:
            print('Hello World!')
        if self:
            serial = []
            queue = [self]
            while queue:
                cur = queue[0]
                if cur:
                    serial.append(cur.val)
                    queue.append(cur.left)
                    queue.append(cur.right)
                else:
                    serial.append('#')
                queue = queue[1:]
            while serial[-1] == '#':
                serial.pop()
            return repr(serial)
        else:
            return None

class Solution(object):

    def generateTrees(self, n):
        if False:
            return 10
        return self.generateTreesRecu(1, n)

    def generateTreesRecu(self, low, high):
        if False:
            print('Hello World!')
        result = []
        if low > high:
            result.append(None)
        for i in xrange(low, high + 1):
            left = self.generateTreesRecu(low, i - 1)
            right = self.generateTreesRecu(i + 1, high)
            for j in left:
                for k in right:
                    cur = TreeNode(i)
                    cur.left = j
                    cur.right = k
                    result.append(cur)
        return result