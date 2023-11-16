import heapq

class Solution(object):

    def findSecondMinimumValue(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def findSecondMinimumValueHelper(root, max_heap, lookup):
            if False:
                while True:
                    i = 10
            if not root:
                return
            if root.val not in lookup:
                heapq.heappush(max_heap, -root.val)
                lookup.add(root.val)
                if len(max_heap) > 2:
                    lookup.remove(-heapq.heappop(max_heap))
            findSecondMinimumValueHelper(root.left, max_heap, lookup)
            findSecondMinimumValueHelper(root.right, max_heap, lookup)
        (max_heap, lookup) = ([], set())
        findSecondMinimumValueHelper(root, max_heap, lookup)
        if len(max_heap) < 2:
            return -1
        return -max_heap[0]