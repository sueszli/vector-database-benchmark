"""
Segment_tree creates a segment tree with a given array and function,
allowing queries to be done later in log(N) time
function takes 2 values and returns a same type value
"""

class SegmentTree:

    def __init__(self, arr, function):
        if False:
            i = 10
            return i + 15
        self.segment = [0 for x in range(3 * len(arr) + 3)]
        self.arr = arr
        self.fn = function
        self.make_tree(0, 0, len(arr) - 1)

    def make_tree(self, i, l, r):
        if False:
            for i in range(10):
                print('nop')
        if l == r:
            self.segment[i] = self.arr[l]
        elif l < r:
            self.make_tree(2 * i + 1, l, int((l + r) / 2))
            self.make_tree(2 * i + 2, int((l + r) / 2) + 1, r)
            self.segment[i] = self.fn(self.segment[2 * i + 1], self.segment[2 * i + 2])

    def __query(self, i, L, R, l, r):
        if False:
            for i in range(10):
                print('nop')
        if l > R or r < L or L > R or (l > r):
            return None
        if L >= l and R <= r:
            return self.segment[i]
        val1 = self.__query(2 * i + 1, L, int((L + R) / 2), l, r)
        val2 = self.__query(2 * i + 2, int((L + R + 2) / 2), R, l, r)
        print(L, R, ' returned ', val1, val2)
        if val1 != None:
            if val2 != None:
                return self.fn(val1, val2)
            return val1
        return val2

    def query(self, L, R):
        if False:
            print('Hello World!')
        return self.__query(0, 0, len(self.arr) - 1, L, R)
'\nExample -\nmytree = SegmentTree([2,4,5,3,4],max)\nmytree.query(2,4)\nmytree.query(0,3) ...\n\nmytree = SegmentTree([4,5,2,3,4,43,3],sum)\nmytree.query(1,8)\n...\n\n'