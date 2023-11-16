import itertools

class Solution(object):

    def minMovesToSeat(self, seats, students):
        if False:
            while True:
                i = 10
        '\n        :type seats: List[int]\n        :type students: List[int]\n        :rtype: int\n        '
        seats.sort()
        students.sort()
        return sum((abs(a - b) for (a, b) in itertools.izip(seats, students)))