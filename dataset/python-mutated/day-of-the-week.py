class Solution(object):

    def dayOfTheWeek(self, day, month, year):
        if False:
            i = 10
            return i + 15
        '\n        :type day: int\n        :type month: int\n        :type year: int\n        :rtype: str\n        '
        DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        if month < 3:
            month += 12
            year -= 1
        (c, y) = divmod(year, 100)
        w = (c // 4 - 2 * c + y + y // 4 + 13 * (month + 1) // 5 + day - 1) % 7
        return DAYS[w]