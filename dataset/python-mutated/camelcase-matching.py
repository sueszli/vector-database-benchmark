class Solution(object):

    def camelMatch(self, queries, pattern):
        if False:
            return 10
        '\n        :type queries: List[str]\n        :type pattern: str\n        :rtype: List[bool]\n        '

        def is_matched(query, pattern):
            if False:
                for i in range(10):
                    print('nop')
            i = 0
            for c in query:
                if i < len(pattern) and pattern[i] == c:
                    i += 1
                elif c.isupper():
                    return False
            return i == len(pattern)
        result = []
        for query in queries:
            result.append(is_matched(query, pattern))
        return result