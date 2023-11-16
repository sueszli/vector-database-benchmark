import sys

class Solution:

    def __init__(self):
        if False:
            print('Hello World!')
        self.mystack = list()
        self.myqueue = list()
        return None

    def pushCharacter(self, char):
        if False:
            print('Hello World!')
        self.mystack.append(char)

    def popCharacter(self):
        if False:
            while True:
                i = 10
        return self.mystack.pop(-1)

    def enqueueCharacter(self, char):
        if False:
            print('Hello World!')
        self.mystack.append(char)

    def dequeueCharacter(self):
        if False:
            i = 10
            return i + 15
        return self.mystack.pop(0)
s = input()
obj = Solution()
l = len(s)
for i in range(l):
    obj.pushCharacter(s[i])
    obj.enqueueCharacter(s[i])
isPalindrome = True
'\npop the top character from stack\ndequeue the first character from queue\ncompare both the characters\n'
for i in range(l // 2):
    if obj.popCharacter() != obj.dequeueCharacter():
        isPalindrome = False
        break
if isPalindrome:
    print('The word, ' + s + ', is a palindrome.')
else:
    print('The word, ' + s + ', is not a palindrome.')