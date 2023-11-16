"""
Topic: 使用slots来减少内存占用
Desc : 
"""

class Date:
    __slots__ = ['year', 'month', 'day']

    def __init__(self, year, month, day):
        if False:
            for i in range(10):
                print('nop')
        self.year = year
        self.month = month
        self.day = day