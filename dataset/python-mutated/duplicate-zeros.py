class Solution(object):

    def duplicateZeros(self, arr):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :rtype: None Do not return anything, modify arr in-place instead.\n        '
        (shift, i) = (0, 0)
        while i + shift < len(arr):
            shift += int(arr[i] == 0)
            i += 1
        i -= 1
        while shift:
            if i + shift < len(arr):
                arr[i + shift] = arr[i]
            if arr[i] == 0:
                shift -= 1
                arr[i + shift] = arr[i]
            i -= 1