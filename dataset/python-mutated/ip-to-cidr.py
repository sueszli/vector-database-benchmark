class Solution(object):

    def ipToCIDR(self, ip, n):
        if False:
            while True:
                i = 10
        '\n        :type ip: str\n        :type n: int\n        :rtype: List[str]\n        '

        def ipToInt(ip):
            if False:
                print('Hello World!')
            result = 0
            for i in ip.split('.'):
                result = 256 * result + int(i)
            return result

        def intToIP(n):
            if False:
                return 10
            return '.'.join((str((n >> i) % 256) for i in (24, 16, 8, 0)))
        start = ipToInt(ip)
        result = []
        while n:
            mask = max(33 - (start & ~(start - 1)).bit_length(), 33 - n.bit_length())
            result.append(intToIP(start) + '/' + str(mask))
            start += 1 << 32 - mask
            n -= 1 << 32 - mask
        return result