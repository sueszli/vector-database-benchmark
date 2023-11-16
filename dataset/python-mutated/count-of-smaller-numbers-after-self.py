class Solution(object):

    def countSmaller(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '

        def countAndMergeSort(num_idxs, start, end, counts):
            if False:
                print('Hello World!')
            if end - start <= 0:
                return
            mid = start + (end - start) // 2
            countAndMergeSort(num_idxs, start, mid, counts)
            countAndMergeSort(num_idxs, mid + 1, end, counts)
            r = mid + 1
            tmp = []
            for i in xrange(start, mid + 1):
                while r <= end and num_idxs[r][0] < num_idxs[i][0]:
                    tmp.append(num_idxs[r])
                    r += 1
                tmp.append(num_idxs[i])
                counts[num_idxs[i][1]] += r - (mid + 1)
            num_idxs[start:start + len(tmp)] = tmp
        num_idxs = []
        counts = [0] * len(nums)
        for (i, num) in enumerate(nums):
            num_idxs.append((num, i))
        countAndMergeSort(num_idxs, 0, len(num_idxs) - 1, counts)
        return counts

class Solution2(object):

    def countSmaller(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '

        class BIT(object):

            def __init__(self, n):
                if False:
                    return 10
                self.__bit = [0] * (n + 1)

            def add(self, i, val):
                if False:
                    while True:
                        i = 10
                i += 1
                while i < len(self.__bit):
                    self.__bit[i] += val
                    i += i & -i

            def query(self, i):
                if False:
                    print('Hello World!')
                i += 1
                ret = 0
                while i > 0:
                    ret += self.__bit[i]
                    i -= i & -i
                return ret
        sorted_nums = sorted(zip(nums, range(len(nums))))
        lookup = {i: new_i for (new_i, (_, i)) in enumerate(sorted_nums)}
        (result, bit) = ([0] * len(nums), BIT(len(nums)))
        for i in reversed(xrange(len(nums))):
            result[i] = bit.query(lookup[i] - 1)
            bit.add(lookup[i], 1)
        return result

class Solution3(object):

    def countSmaller(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        res = [0] * len(nums)
        bst = self.BST()
        for i in reversed(xrange(len(nums))):
            bst.insertNode(nums[i])
            res[i] = bst.query(nums[i])
        return res

    class BST(object):

        class BSTreeNode(object):

            def __init__(self, val):
                if False:
                    for i in range(10):
                        print('nop')
                self.val = val
                self.count = 0
                self.left = self.right = None

        def __init__(self):
            if False:
                return 10
            self.root = None

        def insertNode(self, val):
            if False:
                while True:
                    i = 10
            node = self.BSTreeNode(val)
            if not self.root:
                self.root = node
                return
            curr = self.root
            while curr:
                if node.val < curr.val:
                    curr.count += 1
                    if curr.left:
                        curr = curr.left
                    else:
                        curr.left = node
                        break
                elif curr.right:
                    curr = curr.right
                else:
                    curr.right = node
                    break

        def query(self, val):
            if False:
                i = 10
                return i + 15
            count = 0
            curr = self.root
            while curr:
                if val < curr.val:
                    curr = curr.left
                elif val > curr.val:
                    count += 1 + curr.count
                    curr = curr.right
                else:
                    return count + curr.count
            return 0