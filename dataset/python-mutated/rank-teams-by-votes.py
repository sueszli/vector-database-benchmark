class Solution(object):

    def rankTeams(self, votes):
        if False:
            while True:
                i = 10
        '\n        :type votes: List[str]\n        :rtype: str\n        '
        count = {v: [0] * len(votes[0]) + [v] for v in votes[0]}
        for vote in votes:
            for (i, v) in enumerate(vote):
                count[v][i] -= 1
        return ''.join(sorted(votes[0], key=count.__getitem__))