class Solution(object):

    def verifyPreorder(self, preorder):
        if False:
            for i in range(10):
                print('nop')
        (low, i) = (float('-inf'), -1)
        for p in preorder:
            if p < low:
                return False
            while i >= 0 and p > preorder[i]:
                low = preorder[i]
                i -= 1
            i += 1
            preorder[i] = p
        return True

class Solution2(object):

    def verifyPreorder(self, preorder):
        if False:
            print('Hello World!')
        low = float('-inf')
        path = []
        for p in preorder:
            if p < low:
                return False
            while path and p > path[-1]:
                low = path[-1]
                path.pop()
            path.append(p)
        return True