import collections

class Solution(object):

    def groupAnagrams(self, strs):
        if False:
            print('Hello World!')
        '\n        :type strs: List[str]\n        :rtype: List[List[str]]\n        '
        (anagrams_map, result) = (collections.defaultdict(list), [])
        for s in strs:
            sorted_str = ''.join(sorted(s))
            anagrams_map[sorted_str].append(s)
        for anagram in anagrams_map.values():
            anagram.sort()
            result.append(anagram)
        return result