class Solution(object):

    def wateringPlants(self, plants, capacity):
        if False:
            i = 10
            return i + 15
        '\n        :type plants: List[int]\n        :type capacity: int\n        :rtype: int\n        '
        (result, can) = (len(plants), capacity)
        for (i, x) in enumerate(plants):
            if can < x:
                result += 2 * i
                can = capacity
            can -= x
        return result