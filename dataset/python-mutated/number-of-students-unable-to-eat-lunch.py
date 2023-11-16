import collections

class Solution(object):

    def countStudents(self, students, sandwiches):
        if False:
            while True:
                i = 10
        '\n        :type students: List[int]\n        :type sandwiches: List[int]\n        :rtype: int\n        '
        count = collections.Counter(students)
        for (i, s) in enumerate(sandwiches):
            if not count[s]:
                break
            count[s] -= 1
        else:
            i = len(sandwiches)
        return len(sandwiches) - i