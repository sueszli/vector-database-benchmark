class Solution(object):

    def maskPII(self, S):
        if False:
            while True:
                i = 10
        '\n        :type S: str\n        :rtype: str\n        '
        if '@' in S:
            (first, after) = S.split('@')
            return '{}*****{}@{}'.format(first[0], first[-1], after).lower()
        digits = filter(lambda x: x.isdigit(), S)
        local = '***-***-{}'.format(digits[-4:])
        if len(digits) == 10:
            return local
        return '+{}-{}'.format('*' * (len(digits) - 10), local)