class Solution:

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        if False:
            return 10
        d = {}
        for w in sorted(strs):
            key = tuple(sorted(w))
            d[key] = d.get(key, []) + [w]
        return d.values()