class Solution(object):

    def catchMaximumAmountofPeople(self, team, dist):
        if False:
            while True:
                i = 10
        '\n        :type team: List[int]\n        :type dist: int\n        :rtype: int\n        '
        result = i = j = 0
        while i < len(team) and j < len(team):
            if i + dist < j or team[i] != 1:
                i += 1
            elif j + dist < i or team[j] != 0:
                j += 1
            else:
                result += 1
                i += 1
                j += 1
        return result

class Solution2(object):

    def catchMaximumAmountofPeople(self, team, dist):
        if False:
            i = 10
            return i + 15
        '\n        :type team: List[int]\n        :type dist: int\n        :rtype: int\n        '
        result = j = 0
        for i in xrange(len(team)):
            if not team[i]:
                continue
            while j < i - dist:
                j += 1
            while j <= min(i + dist, len(team) - 1):
                if team[j] == 0:
                    break
                j += 1
            if j <= min(i + dist, len(team) - 1):
                result += 1
                j += 1
        return result