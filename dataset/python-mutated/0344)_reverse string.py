class Solution:

    def reverseString(self, s: List[str]) -> None:
        if False:
            while True:
                i = 10
        '\n        Do not return anything, modify s in-place instead.\n        '
        st = 0
        end = len(s) - 1
        while end > st:
            tp1 = s[st]
            tp2 = s[end]
            s[st] = tp2
            s[end] = tp1
            st = st + 1
            end = end - 1