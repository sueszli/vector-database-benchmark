import collections

class Solution(object):

    def watchedVideosByFriends(self, watchedVideos, friends, id, level):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type watchedVideos: List[List[str]]\n        :type friends: List[List[int]]\n        :type id: int\n        :type level: int\n        :rtype: List[str]\n        '
        (curr_level, lookup) = (set([id]), set([id]))
        for _ in xrange(level):
            curr_level = set((j for i in curr_level for j in friends[i] if j not in lookup))
            lookup |= curr_level
        count = collections.Counter([v for i in curr_level for v in watchedVideos[i]])
        return sorted(count.keys(), key=lambda x: (count[x], x))