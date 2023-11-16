class SegTreeNode:

    def __init__(self, val=0):
        if False:
            return 10
        self.left = -1
        self.right = -1
        self.val = val

class SegmentTree:

    def __init__(self, nums, function):
        if False:
            i = 10
            return i + 15
        self.size = len(nums)
        self.tree = [SegTreeNode() for _ in range(4 * self.size)]
        self.nums = nums
        self.function = function
        if self.size > 0:
            self.__build(0, 0, self.size - 1)

    def update_point(self, i, val):
        if False:
            i = 10
            return i + 15
        self.nums[i] = val
        self.__update_point(i, val, 0)

    def query_interval(self, q_left, q_right):
        if False:
            i = 10
            return i + 15
        return self.__query_interval(q_left, q_right, 0)

    def get_nums(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(self.size):
            self.nums[i] = self.query_interval(i, i)
        return self.nums

    def __build(self, index, left, right):
        if False:
            return 10
        self.tree[index].left = left
        self.tree[index].right = right
        if left == right:
            self.tree[index].val = self.nums[left]
            return
        mid = left + (right - left) // 2
        left_index = index * 2 + 1
        right_index = index * 2 + 2
        self.__build(left_index, left, mid)
        self.__build(right_index, mid + 1, right)
        self.__pushup(index)

    def __update_point(self, i, val, index):
        if False:
            while True:
                i = 10
        left = self.tree[index].left
        right = self.tree[index].right
        if left == right:
            self.tree[index].val = val
            return
        mid = left + (right - left) // 2
        left_index = index * 2 + 1
        right_index = index * 2 + 2
        if i <= mid:
            self.__update_point(i, val, left_index)
        else:
            self.__update_point(i, val, right_index)
        self.__pushup(index)

    def __query_interval(self, q_left, q_right, index):
        if False:
            while True:
                i = 10
        left = self.tree[index].left
        right = self.tree[index].right
        if left >= q_left and right <= q_right:
            return self.tree[index].val
        if right < q_left or left > q_right:
            return 0
        mid = left + (right - left) // 2
        left_index = index * 2 + 1
        right_index = index * 2 + 2
        res_left = 0
        res_right = 0
        if q_left <= mid:
            res_left = self.__query_interval(q_left, q_right, left_index)
        if q_right > mid:
            res_right = self.__query_interval(q_left, q_right, right_index)
        return self.function(res_left, res_right)

    def __pushup(self, index):
        if False:
            return 10
        left_index = index * 2 + 1
        right_index = index * 2 + 2
        self.tree[index].val = self.function(self.tree[left_index].val, self.tree[right_index].val)

class Solution:

    def __init__(self, nums: List[int]):
        if False:
            for i in range(10):
                print('nop')
        self.STree = SegmentTree(nums, lambda x, y: x + y)

    def update(self, index: int, val: int) -> None:
        if False:
            i = 10
            return i + 15
        self.STree.update_point(index, val)

    def sumRange(self, left: int, right: int) -> int:
        if False:
            while True:
                i = 10
        return self.STree.query_interval(left, right)