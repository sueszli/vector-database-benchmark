class Solution(object):

    def reverseString(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: List[str]\n        :rtype: None Do not return anything, modify s in-place instead.\n        '
        (i, j) = (0, len(s) - 1)
        while i < j:
            (s[i], s[j]) = (s[j], s[i])
            i += 1
            j -= 1