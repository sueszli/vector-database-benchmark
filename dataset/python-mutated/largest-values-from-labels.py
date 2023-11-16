import collections

class Solution(object):

    def largestValsFromLabels(self, values, labels, num_wanted, use_limit):
        if False:
            print('Hello World!')
        '\n        :type values: List[int]\n        :type labels: List[int]\n        :type num_wanted: int\n        :type use_limit: int\n        :rtype: int\n        '
        counts = collections.defaultdict(int)
        val_labs = zip(values, labels)
        val_labs.sort(reverse=True)
        result = 0
        for (val, lab) in val_labs:
            if counts[lab] >= use_limit:
                continue
            result += val
            counts[lab] += 1
            num_wanted -= 1
            if num_wanted == 0:
                break
        return result