class Solution(object):

    def findOriginalArray(self, changed):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type changed: List[int]\n        :rtype: List[int]\n        '
        if len(changed) % 2:
            return []
        cnts = collections.Counter(changed)
        for x in sorted(cnts.iterkeys()):
            if cnts[x] > cnts[2 * x]:
                return []
            cnts[2 * x] -= cnts[x] if x else cnts[x] // 2
        return list(cnts.elements())