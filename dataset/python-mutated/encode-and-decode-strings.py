class Codec(object):

    def encode(self, strs):
        if False:
            for i in range(10):
                print('nop')
        'Encodes a list of strings to a single string.\n\n        :type strs: List[str]\n        :rtype: str\n        '
        encoded_str = ''
        for s in strs:
            encoded_str += '%0*x' % (8, len(s)) + s
        return encoded_str

    def decode(self, s):
        if False:
            while True:
                i = 10
        'Decodes a single string to a list of strings.\n\n        :type s: str\n        :rtype: List[str]\n        '
        i = 0
        strs = []
        while i < len(s):
            l = int(s[i:i + 8], 16)
            strs.append(s[i + 8:i + 8 + l])
            i += 8 + l
        return strs