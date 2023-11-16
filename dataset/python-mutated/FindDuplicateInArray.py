from collections import Counter

class Solution:

    def repeatedNumber(self, A):
        if False:
            while True:
                i = 10
        c = Counter(A)
        for i in c:
            if c[i] > 1:
                return i
        return -1