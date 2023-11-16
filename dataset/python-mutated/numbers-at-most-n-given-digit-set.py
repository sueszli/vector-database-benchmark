class Solution(object):

    def atMostNGivenDigitSet(self, D, N):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type D: List[str]\n        :type N: int\n        :rtype: int\n        '
        str_N = str(N)
        set_D = set(D)
        result = sum((len(D) ** i for i in xrange(1, len(str_N))))
        i = 0
        while i < len(str_N):
            result += sum((c < str_N[i] for c in D)) * len(D) ** (len(str_N) - i - 1)
            if str_N[i] not in set_D:
                break
            i += 1
        return result + int(i == len(str_N))