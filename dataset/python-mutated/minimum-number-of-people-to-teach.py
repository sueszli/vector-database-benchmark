import collections

class Solution(object):

    def minimumTeachings(self, n, languages, friendships):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type languages: List[List[int]]\n        :type friendships: List[List[int]]\n        :rtype: int\n        '
        language_sets = map(set, languages)
        candidates = set((i - 1 for (u, v) in friendships if not language_sets[u - 1] & language_sets[v - 1] for i in [u, v]))
        count = collections.Counter()
        for i in candidates:
            count += collections.Counter(languages[i])
        return len(candidates) - max(count.values() + [0])