class Solution(object):

    def lengthLongestPath(self, input):
        if False:
            while True:
                i = 10
        '\n        :type input: str\n        :rtype: int\n        '

        def split_iter(s, tok):
            if False:
                for i in range(10):
                    print('nop')
            start = 0
            for i in xrange(len(s)):
                if s[i] == tok:
                    yield s[start:i]
                    start = i + 1
            yield s[start:]
        max_len = 0
        path_len = {0: 0}
        for line in split_iter(input, '\n'):
            name = line.lstrip('\t')
            depth = len(line) - len(name)
            if '.' in name:
                max_len = max(max_len, path_len[depth] + len(name))
            else:
                path_len[depth + 1] = path_len[depth] + len(name) + 1
        return max_len