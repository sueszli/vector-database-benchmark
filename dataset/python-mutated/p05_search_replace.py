"""
Topic: 字符串搜索和替换
Desc : 
"""
import re
from calendar import month_abbr

def change_date(m):
    if False:
        while True:
            i = 10
    mon_name = month_abbr[int(m.group(1))]
    return '{} {} {}'.format(m.group(2), mon_name, m.group(3))

def search_replace():
    if False:
        return 10
    text = 'yeah, but no, but yeah, but no, but yeah'
    print(text.replace('yeah', 'yep'))
    text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
    print(re.sub('(\\d+)/(\\d+)/(\\d+)', '\\3-\\1-\\2', text))
    print(re.sub('(?P<month>\\d+)/(?P<day>\\d+)/(?P<year>\\d+)', '\\g<year>-\\g<month>-\\g<day>', text))
    datepat = re.compile('(\\d+)/(\\d+)/(\\d+)')
    print(datepat.sub('\\3-\\1-\\2', text))
    print(datepat.sub(change_date, text))
    (newtext, n) = datepat.subn('\\3-\\1-\\2', text)
    print(newtext, n)
if __name__ == '__main__':
    search_replace()