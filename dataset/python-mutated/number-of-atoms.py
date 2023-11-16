import collections
import re

class Solution(object):

    def countOfAtoms(self, formula):
        if False:
            i = 10
            return i + 15
        '\n        :type formula: str\n        :rtype: str\n        '
        parse = re.findall('([A-Z][a-z]*)(\\d*)|(\\()|(\\))(\\d*)', formula)
        stk = [collections.Counter()]
        for (name, m1, left_open, right_open, m2) in parse:
            if name:
                stk[-1][name] += int(m1 or 1)
            if left_open:
                stk.append(collections.Counter())
            if right_open:
                top = stk.pop()
                for (k, v) in top.iteritems():
                    stk[-1][k] += v * int(m2 or 1)
        return ''.join((name + (str(stk[-1][name]) if stk[-1][name] > 1 else '') for name in sorted(stk[-1])))