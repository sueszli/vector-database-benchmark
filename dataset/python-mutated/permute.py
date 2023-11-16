"""
Given a collection of distinct numbers, return all possible permutations.

For example,
[1,2,3] have the following permutations:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
"""

def permute(elements):
    if False:
        print('Hello World!')
    '\n        returns a list with the permuations.\n    '
    if len(elements) <= 1:
        return [elements]
    else:
        tmp = []
        for perm in permute(elements[1:]):
            for i in range(len(elements)):
                tmp.append(perm[:i] + elements[0:1] + perm[i:])
        return tmp

def permute_iter(elements):
    if False:
        print('Hello World!')
    '\n        iterator: returns a perumation by each call.\n    '
    if len(elements) <= 1:
        yield elements
    else:
        for perm in permute_iter(elements[1:]):
            for i in range(len(elements)):
                yield (perm[:i] + elements[0:1] + perm[i:])

def permute_recursive(nums):
    if False:
        for i in range(10):
            print('nop')

    def dfs(res, nums, path):
        if False:
            i = 10
            return i + 15
        if not nums:
            res.append(path)
        for i in range(len(nums)):
            print(nums[:i] + nums[i + 1:])
            dfs(res, nums[:i] + nums[i + 1:], path + [nums[i]])
    res = []
    dfs(res, nums, [])
    return res