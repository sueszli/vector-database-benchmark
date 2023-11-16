class Solution(object):

    def splitMessage(self, message, limit):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type message: str\n        :type limit: int\n        :rtype: List[str]\n        '
        (cnt, l, total, base) = (1, 1, len(message) + 1, 1)
        while 3 + l * 2 < limit:
            if total + (3 + l) * cnt <= limit * cnt:
                break
            cnt += 1
            if cnt == base * 10:
                l += 1
                base *= 10
            total += l
        if 3 + l * 2 >= limit:
            return []
        result = []
        j = 0
        for i in xrange(cnt):
            l = limit - (3 + len(str(i + 1)) + len(str(cnt)))
            result.append('%s<%s/%s>' % (message[j:j + l], i + 1, cnt))
            j += l
        return result