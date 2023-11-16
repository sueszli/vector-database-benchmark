class Solution(object):

    def evaluate(self, expression):
        if False:
            print('Hello World!')
        '\n        :type expression: str\n        :rtype: int\n        '

        def getval(lookup, x):
            if False:
                for i in range(10):
                    print('nop')
            return lookup.get(x, x)

        def evaluate(tokens, lookup):
            if False:
                for i in range(10):
                    print('nop')
            if tokens[0] in ('add', 'mult'):
                (a, b) = map(int, map(lambda x: getval(lookup, x), tokens[1:]))
                return str(a + b if tokens[0] == 'add' else a * b)
            for i in xrange(1, len(tokens) - 1, 2):
                if tokens[i + 1]:
                    lookup[tokens[i]] = getval(lookup, tokens[i + 1])
            return getval(lookup, tokens[-1])
        (tokens, lookup, stk) = ([''], {}, [])
        for c in expression:
            if c == '(':
                if tokens[0] == 'let':
                    evaluate(tokens, lookup)
                stk.append((tokens, dict(lookup)))
                tokens = ['']
            elif c == ' ':
                tokens.append('')
            elif c == ')':
                val = evaluate(tokens, lookup)
                (tokens, lookup) = stk.pop()
                tokens[-1] += val
            else:
                tokens[-1] += c
        return int(tokens[0])