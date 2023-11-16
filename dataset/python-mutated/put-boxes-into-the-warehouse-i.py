class Solution(object):

    def maxBoxesInWarehouse(self, boxes, warehouse):
        if False:
            while True:
                i = 10
        '\n        :type boxes: List[int]\n        :type warehouse: List[int]\n        :rtype: int\n        '
        boxes.sort(reverse=True)
        result = 0
        for h in boxes:
            if h > warehouse[result]:
                continue
            result += 1
            if result == len(warehouse):
                break
        return result

class Solution2(object):

    def maxBoxesInWarehouse(self, boxes, warehouse):
        if False:
            print('Hello World!')
        '\n        :type boxes: List[int]\n        :type warehouse: List[int]\n        :rtype: int\n        '
        boxes.sort()
        for i in xrange(1, len(warehouse)):
            warehouse[i] = min(warehouse[i], warehouse[i - 1])
        (result, curr) = (0, 0)
        for h in reversed(warehouse):
            if boxes[curr] > h:
                continue
            result += 1
            curr += 1
            if curr == len(boxes):
                break
        return result