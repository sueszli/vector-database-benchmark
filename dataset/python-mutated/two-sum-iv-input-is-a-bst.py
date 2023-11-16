class Solution(object):

    def findTarget(self, root, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :type k: int\n        :rtype: bool\n        '

        class BSTIterator(object):

            def __init__(self, root, forward):
                if False:
                    return 10
                self.__node = root
                self.__forward = forward
                self.__s = []
                self.__cur = None
                self.next()

            def val(self):
                if False:
                    return 10
                return self.__cur

            def next(self):
                if False:
                    while True:
                        i = 10
                while self.__node or self.__s:
                    if self.__node:
                        self.__s.append(self.__node)
                        self.__node = self.__node.left if self.__forward else self.__node.right
                    else:
                        self.__node = self.__s.pop()
                        self.__cur = self.__node.val
                        self.__node = self.__node.right if self.__forward else self.__node.left
                        break
        if not root:
            return False
        (left, right) = (BSTIterator(root, True), BSTIterator(root, False))
        while left.val() < right.val():
            if left.val() + right.val() == k:
                return True
            elif left.val() + right.val() < k:
                left.next()
            else:
                right.next()
        return False