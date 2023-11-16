class Solution(object):

    def canMakeArithmeticProgression(self, arr):
        if False:
            print('Hello World!')
        '\n        :type arr: List[int]\n        :rtype: bool\n        '
        m = min(arr)
        d = (max(arr) - m) // (len(arr) - 1)
        if not d:
            return True
        i = 0
        while i < len(arr):
            if arr[i] == m + i * d:
                i += 1
            else:
                (j, r) = divmod(arr[i] - m, d)
                if r or j >= len(arr) or arr[i] == arr[j]:
                    return False
                (arr[i], arr[j]) = (arr[j], arr[i])
        return True