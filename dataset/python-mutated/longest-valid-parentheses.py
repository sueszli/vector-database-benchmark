class Solution(object):

    def longestValidParentheses(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: int\n        '

        def length(it, start, c):
            if False:
                for i in range(10):
                    print('nop')
            (depth, longest) = (0, 0)
            for i in it:
                if s[i] == c:
                    depth += 1
                else:
                    depth -= 1
                    if depth < 0:
                        (start, depth) = (i, 0)
                    elif depth == 0:
                        longest = max(longest, abs(i - start))
            return longest
        return max(length(xrange(len(s)), -1, '('), length(reversed(xrange(len(s))), len(s), ')'))

class Solution2(object):

    def longestValidParentheses(self, s):
        if False:
            for i in range(10):
                print('nop')
        (longest, last, indices) = (0, -1, [])
        for i in xrange(len(s)):
            if s[i] == '(':
                indices.append(i)
            elif not indices:
                last = i
            else:
                indices.pop()
                if not indices:
                    longest = max(longest, i - last)
                else:
                    longest = max(longest, i - indices[-1])
        return longest