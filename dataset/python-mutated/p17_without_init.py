"""
Topic: 不使用__init__方法初创建对象
Desc : 
"""

class Date:

    def __init__(self, year, month, day):
        if False:
            for i in range(10):
                print('nop')
        self.year = year
        self.month = month
        self.day = day
data = {'year': 2012, 'month': 8, 'day': 29}
d = Date.__new__(Date)
for (key, value) in data.items():
    setattr(d, key, value)
print(d.year, d.month, d.day)
from time import localtime

class Date:

    def __init__(self, year, month, day):
        if False:
            i = 10
            return i + 15
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def today(cls):
        if False:
            return 10
        d = cls.__new__(cls)
        t = localtime()
        d.year = t.tm_year
        d.month = t.tm_mon
        d.day = t.tm_mday
        return d