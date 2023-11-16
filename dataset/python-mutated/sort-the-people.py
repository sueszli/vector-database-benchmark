class Solution(object):

    def sortPeople(self, names, heights):
        if False:
            i = 10
            return i + 15
        '\n        :type names: List[str]\n        :type heights: List[int]\n        :rtype: List[str]\n        '
        order = range(len(names))
        order.sort(key=lambda x: heights[x], reverse=True)
        return [names[i] for i in order]