import collections

class Solution(object):

    def getFolderNames(self, names):
        if False:
            while True:
                i = 10
        '\n        :type names: List[str]\n        :rtype: List[str]\n        '
        count = collections.Counter()
        (result, lookup) = ([], set())
        for name in names:
            while True:
                name_with_suffix = '{}({})'.format(name, count[name]) if count[name] else name
                count[name] += 1
                if name_with_suffix not in lookup:
                    break
            result.append(name_with_suffix)
            lookup.add(name_with_suffix)
        return result