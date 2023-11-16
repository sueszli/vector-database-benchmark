class Solution(object):

    def survivedRobotsHealths(self, positions, healths, directions):
        if False:
            print('Hello World!')
        '\n        :type positions: List[int]\n        :type healths: List[int]\n        :type directions: str\n        :rtype: List[int]\n        '
        stk = []
        for i in sorted(xrange(len(positions)), key=lambda x: positions[x]):
            if directions[i] == 'R':
                stk.append(i)
                continue
            while stk:
                if healths[stk[-1]] == healths[i]:
                    healths[stk.pop()] = healths[i] = 0
                    break
                if healths[stk[-1]] > healths[i]:
                    healths[i] = 0
                    healths[stk[-1]] -= 1
                    break
                healths[stk.pop()] = 0
                healths[i] -= 1
        return [x for x in healths if x]