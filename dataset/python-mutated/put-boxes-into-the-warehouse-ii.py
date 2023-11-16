class Solution(object):

    def maxBoxesInWarehouse(self, boxes, warehouse):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type boxes: List[int]\n        :type warehouse: List[int]\n        :rtype: int\n        '
        boxes.sort(reverse=True)
        (left, right) = (0, len(warehouse) - 1)
        for h in boxes:
            if h <= warehouse[left]:
                left += 1
            elif h <= warehouse[right]:
                right -= 1
            if left > right:
                break
        return left + (len(warehouse) - 1 - right)