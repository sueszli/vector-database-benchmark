class Solution(object):

    def evaluate(self, s, knowledge):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type knowledge: List[List[str]]\n        :rtype: str\n        '
        lookup = {k: v for (k, v) in knowledge}
        (result, curr) = ([], [])
        has_pair = False
        for c in s:
            if c == '(':
                has_pair = True
            elif c == ')':
                has_pair = False
                result.append(lookup.get(''.join(curr), '?'))
                curr = []
            elif has_pair:
                curr.append(c)
            else:
                result.append(c)
        return ''.join(result)