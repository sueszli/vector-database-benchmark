class Solution(object):

    def arraysIntersection(self, arr1, arr2, arr3):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr1: List[int]\n        :type arr2: List[int]\n        :type arr3: List[int]\n        :rtype: List[int]\n        '
        result = []
        (i, j, k) = (0, 0, 0)
        while i != len(arr1) and j != len(arr2) and (k != len(arr3)):
            if arr1[i] == arr2[j] == arr3[k]:
                result.append(arr1[i])
                i += 1
                j += 1
                k += 1
            else:
                curr = max(arr1[i], arr2[j], arr3[k])
                while i != len(arr1) and arr1[i] < curr:
                    i += 1
                while j != len(arr2) and arr2[j] < curr:
                    j += 1
                while k != len(arr3) and arr3[k] < curr:
                    k += 1
        return result

class Solution2(object):

    def arraysIntersection(self, arr1, arr2, arr3):
        if False:
            print('Hello World!')
        '\n        :type arr1: List[int]\n        :type arr2: List[int]\n        :type arr3: List[int]\n        :rtype: List[int]\n        '
        intersect = reduce(set.intersection, map(set, [arr2, arr3]))
        return [x for x in arr1 if x in intersect]