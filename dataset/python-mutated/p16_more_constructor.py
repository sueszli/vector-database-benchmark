"""
Topic: 定义类的多个构造函数
Desc : 
"""
import time

class Date:
    """方法一：使用类方法"""

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
            print('Hello World!')
        t = time.localtime()
        return cls(t.tm_year, t.tm_mon, t.tm_mday)
a = Date(2012, 12, 21)
b = Date.today()