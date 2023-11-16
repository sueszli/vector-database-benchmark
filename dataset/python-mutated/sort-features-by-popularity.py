import collections

class Solution(object):

    def sortFeatures(self, features, responses):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type features: List[str]\n        :type responses: List[str]\n        :rtype: List[str]\n        '
        features_set = set(features)
        order = {word: i for (i, word) in enumerate(features)}
        freq = collections.defaultdict(int)
        for r in responses:
            for word in set(r.split(' ')):
                if word in features_set:
                    freq[word] += 1
        features.sort(key=lambda x: (-freq[x], order[x]))
        return features