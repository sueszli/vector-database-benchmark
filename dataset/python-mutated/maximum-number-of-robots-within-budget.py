import collections

class Solution(object):

    def maximumRobots(self, chargeTimes, runningCosts, budget):
        if False:
            i = 10
            return i + 15
        '\n        :type chargeTimes: List[int]\n        :type runningCosts: List[int]\n        :type budget: int\n        :rtype: int\n        '
        result = left = curr = 0
        dq = collections.deque()
        for right in xrange(len(chargeTimes)):
            while dq and chargeTimes[dq[-1]] <= chargeTimes[right]:
                dq.pop()
            dq.append(right)
            curr += runningCosts[right]
            if chargeTimes[dq[0]] + (right - left + 1) * curr > budget:
                if dq[0] == left:
                    dq.popleft()
                curr -= runningCosts[left]
                left += 1
        return right - left + 1
import collections

class Solution2(object):

    def maximumRobots(self, chargeTimes, runningCosts, budget):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type chargeTimes: List[int]\n        :type runningCosts: List[int]\n        :type budget: int\n        :rtype: int\n        '
        result = left = curr = 0
        dq = collections.deque()
        for right in xrange(len(chargeTimes)):
            while dq and chargeTimes[dq[-1]] <= chargeTimes[right]:
                dq.pop()
            dq.append(right)
            curr += runningCosts[right]
            while dq and chargeTimes[dq[0]] + (right - left + 1) * curr > budget:
                if dq[0] == left:
                    dq.popleft()
                curr -= runningCosts[left]
                left += 1
            result = max(result, right - left + 1)
        return result