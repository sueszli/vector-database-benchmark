class Solution(object):

    def maximumUnits(self, boxTypes, truckSize):
        if False:
            print('Hello World!')
        '\n        :type boxTypes: List[List[int]]\n        :type truckSize: int\n        :rtype: int\n        '
        boxTypes.sort(key=lambda x: x[1], reverse=True)
        result = 0
        for (box, units) in boxTypes:
            if truckSize > box:
                truckSize -= box
                result += box * units
            else:
                result += truckSize * units
                break
        return result