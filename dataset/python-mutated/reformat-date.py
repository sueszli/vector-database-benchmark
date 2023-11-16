class Solution(object):

    def reformatDate(self, date):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type date: str\n        :rtype: str\n        '
        lookup = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        return '{:04d}-{:02d}-{:02d}'.format(int(date[-4:]), lookup[date[-8:-5]], int(date[:date.index(' ') - 2]))