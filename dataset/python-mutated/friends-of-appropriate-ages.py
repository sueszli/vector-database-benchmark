import collections

class Solution(object):

    def numFriendRequests(self, ages):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type ages: List[int]\n        :rtype: int\n        '

        def request(a, b):
            if False:
                i = 10
                return i + 15
            return 0.5 * a + 7 < b <= a
        c = collections.Counter(ages)
        return sum((int(request(a, b)) * c[a] * (c[b] - int(a == b)) for a in c for b in c))