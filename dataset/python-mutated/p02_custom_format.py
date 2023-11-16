"""
Topic: 对象自定义格式化
Desc : 
"""
_formats = {'ymd': '{d.year}-{d.month}-{d.day}', 'mdy': '{d.month}/{d.day}/{d.year}', 'dmy': '{d.day}/{d.month}/{d.year}'}

class Date:

    def __init__(self, year, month, day):
        if False:
            return 10
        self.year = year
        self.month = month
        self.day = day

    def __format__(self, code):
        if False:
            for i in range(10):
                print('nop')
        if code == '':
            code = 'ymd'
        fmt = _formats[code]
        return fmt.format(d=self)
d = Date(2012, 12, 21)
print(d)
print(format(d, 'mdy'))
print('The date is {:ymd}'.format(d))
print('The date is {:mdy}'.format(d))