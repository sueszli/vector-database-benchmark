class Solution(object):

    def lengthOfLastWord(self, s):
        if False:
            for i in range(10):
                print('nop')
        length = 0
        for i in reversed(s):
            if i == ' ':
                if length:
                    break
            else:
                length += 1
        return length

class Solution2(object):

    def lengthOfLastWord(self, s):
        if False:
            print('Hello World!')
        return len(s.strip().split(' ')[-1])