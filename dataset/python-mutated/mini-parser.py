class NestedInteger(object):

    def __init__(self, value=None):
        if False:
            return 10
        '\n       If value is not specified, initializes an empty list.\n       Otherwise initializes a single integer equal to value.\n       '

    def isInteger(self):
        if False:
            print('Hello World!')
        '\n       @return True if this NestedInteger holds a single integer, rather than a nested list.\n       :rtype bool\n       '

    def add(self, elem):
        if False:
            while True:
                i = 10
        '\n       Set this NestedInteger to hold a nested list and adds a nested integer elem to it.\n       :rtype void\n       '

    def setInteger(self, value):
        if False:
            i = 10
            return i + 15
        '\n       Set this NestedInteger to hold a single integer equal to value.\n       :rtype void\n       '

    def getInteger(self):
        if False:
            for i in range(10):
                print('nop')
        '\n       @return the single integer that this NestedInteger holds, if it holds a single integer\n       Return None if this NestedInteger holds a nested list\n       :rtype int\n       '

    def getList(self):
        if False:
            for i in range(10):
                print('nop')
        '\n       @return the nested list that this NestedInteger holds, if it holds a nested list\n       Return None if this NestedInteger holds a single integer\n       :rtype List[NestedInteger]\n       '

class Solution(object):

    def deserialize(self, s):
        if False:
            for i in range(10):
                print('nop')
        if not s:
            return NestedInteger()
        if s[0] != '[':
            return NestedInteger(int(s))
        stk = []
        i = 0
        for j in xrange(len(s)):
            if s[j] == '[':
                stk += (NestedInteger(),)
                i = j + 1
            elif s[j] in ',]':
                if s[j - 1].isdigit():
                    stk[-1].add(NestedInteger(int(s[i:j])))
                if s[j] == ']' and len(stk) > 1:
                    cur = stk[-1]
                    stk.pop()
                    stk[-1].add(cur)
                i = j + 1
        return stk[-1]