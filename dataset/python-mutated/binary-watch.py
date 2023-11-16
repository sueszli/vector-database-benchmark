class Solution(object):

    def readBinaryWatch(self, num):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type num: int\n        :rtype: List[str]\n        '

        def bit_count(bits):
            if False:
                for i in range(10):
                    print('nop')
            count = 0
            while bits:
                bits &= bits - 1
                count += 1
            return count
        return ['%d:%02d' % (h, m) for h in xrange(12) for m in xrange(60) if bit_count(h) + bit_count(m) == num]

    def readBinaryWatch2(self, num):
        if False:
            i = 10
            return i + 15
        '\n        :type num: int\n        :rtype: List[str]\n        '
        return ['{0}:{1}'.format(str(h), str(m).zfill(2)) for h in range(12) for m in range(60) if (bin(h) + bin(m)).count('1') == num]