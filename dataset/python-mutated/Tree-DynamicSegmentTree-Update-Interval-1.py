class SegTreeNode:

    def __init__(self, left=-1, right=-1, val=False, lazy_tag=None, leftNode=None, rightNode=None):
        if False:
            while True:
                i = 10
        self.left = left
        self.right = right
        self.mid = left + (right - left) // 2
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.val = val
        self.lazy_tag = lazy_tag

class SegmentTree:

    def __init__(self, function):
        if False:
            return 10
        self.tree = SegTreeNode(0, int(1000000000.0))
        self.function = function

    def update_point(self, i, val):
        if False:
            return 10
        self.__update_point(i, val, self.tree)

    def update_interval(self, q_left, q_right, val):
        if False:
            while True:
                i = 10
        self.__update_interval(q_left, q_right, val, self.tree)

    def query_interval(self, q_left, q_right):
        if False:
            print('Hello World!')
        return self.__query_interval(q_left, q_right, self.tree)

    def get_nums(self, length):
        if False:
            for i in range(10):
                print('nop')
        nums = [0 for _ in range(length)]
        for i in range(length):
            nums[i] = self.query_interval(i, i)
        return nums

    def __update_point(self, i, val, node):
        if False:
            print('Hello World!')
        if node.left == node.right:
            node.val = val
            return
        if i <= node.mid:
            self.__update_point(i, val, node.leftNode)
        else:
            self.__update_point(i, val, node.rightNode)
        self.__pushup(node)

    def __update_interval(self, q_left, q_right, val, node):
        if False:
            i = 10
            return i + 15
        if node.left >= q_left and node.right <= q_right:
            node.lazy_tag = val
            interval_size = node.right - node.left + 1
            node.val = val * interval_size
            return
        if node.right < q_left or node.left > q_right:
            return
        self.__pushdown(node)
        if q_left <= node.mid:
            self.__update_interval(q_left, q_right, val, node.leftNode)
        if q_right > node.mid:
            self.__update_interval(q_left, q_right, val, node.rightNode)
        self.__pushup(node)

    def __query_interval(self, q_left, q_right, node):
        if False:
            print('Hello World!')
        if node.left >= q_left and node.right <= q_right:
            return node.val
        if node.right < q_left or node.left > q_right:
            return 0
        self.__pushdown(node)
        res_left = 0
        res_right = 0
        if q_left <= node.mid:
            res_left = self.__query_interval(q_left, q_right, node.leftNode)
        if q_right > node.mid:
            res_right = self.__query_interval(q_left, q_right, node.rightNode)
        return self.function(res_left, res_right)

    def __pushup(self, node):
        if False:
            i = 10
            return i + 15
        if node.leftNode and node.rightNode:
            node.val = self.function(node.leftNode.val, node.rightNode.val)

    def __pushdown(self, node):
        if False:
            print('Hello World!')
        if node.leftNode is None:
            node.leftNode = SegTreeNode(node.left, node.mid)
        if node.rightNode is None:
            node.rightNode = SegTreeNode(node.mid + 1, node.right)
        lazy_tag = node.lazy_tag
        if node.lazy_tag is None:
            return
        node.leftNode.lazy_tag = lazy_tag
        left_size = node.leftNode.right - node.leftNode.left + 1
        node.leftNode.val = lazy_tag * left_size
        node.rightNode.lazy_tag = lazy_tag
        right_size = node.rightNode.right - node.rightNode.left + 1
        node.rightNode.val = lazy_tag * right_size
        node.lazy_tag = None

class Solution:

    def __init__(self):
        if False:
            return 10
        self.STree = SegmentTree(lambda x, y: x + y)

    def update(self, left: int, right: int, val) -> None:
        if False:
            return 10
        self.STree.update_interval(left, right, val)

    def sumRange(self, left: int, right: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.STree.query_interval(left, right)