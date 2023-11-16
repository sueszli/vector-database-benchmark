import collections

class Solution(object):

    def groupThePeople(self, groupSizes):
        if False:
            return 10
        '\n        :type groupSizes: List[int]\n        :rtype: List[List[int]]\n        '
        (groups, result) = (collections.defaultdict(list), [])
        for (i, size) in enumerate(groupSizes):
            groups[size].append(i)
            if len(groups[size]) == size:
                result.append(groups.pop(size))
        return result