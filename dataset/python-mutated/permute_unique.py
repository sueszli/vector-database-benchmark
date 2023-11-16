"""
Given a collection of numbers that might contain duplicates,
return all possible unique permutations.

For example,
[1,1,2] have the following unique permutations:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]
"""

def permute_unique(nums):
    if False:
        for i in range(10):
            print('nop')
    perms = [[]]
    for n in nums:
        new_perms = []
        for l in perms:
            for i in range(len(l) + 1):
                new_perms.append(l[:i] + [n] + l[i:])
                if i < len(l) and l[i] == n:
                    break
        perms = new_perms
    return perms