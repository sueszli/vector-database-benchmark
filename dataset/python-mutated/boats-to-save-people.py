class Solution(object):

    def numRescueBoats(self, people, limit):
        if False:
            i = 10
            return i + 15
        '\n        :type people: List[int]\n        :type limit: int\n        :rtype: int\n        '
        people.sort()
        result = 0
        (left, right) = (0, len(people) - 1)
        while left <= right:
            result += 1
            if people[left] + people[right] <= limit:
                left += 1
            right -= 1
        return result