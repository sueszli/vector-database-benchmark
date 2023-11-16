"""
Topic: 字符串匹配和搜索
Desc : 
"""
import re

def match_search():
    if False:
        while True:
            i = 10
    text = 'yeah, but no, but yeah, but no, but yeah'
    print(text == 'yeah')
    print(text.startswith('yeah'))
    print(text.endswith('no'))
    print(text.find('no'))
    text1 = '11/27/2012'
    text2 = 'Nov 27, 2012'
    if re.match('\\d+/\\d+/\\d+', text1):
        print('yes')
    else:
        print('no')
    if re.match('\\d+/\\d+/\\d+', text2):
        print('yes')
    else:
        print('no')
    datepat = re.compile('\\d+/\\d+/\\d+')
    if datepat.match(text1):
        print('yes')
    else:
        print('no')
    if datepat.match(text2):
        print('yes')
    else:
        print('no')
    text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
    print(datepat.findall(text))
    datepat = re.compile('(\\d+)/(\\d+)/(\\d+)')
    m = datepat.match('11/27/2012')
    print(m.group(0))
    print(m.group(1))
    print(m.group(2))
    print(m.group(3))
    print(m.groups())
    (month, day, year) = m.groups()
    print(datepat.findall(text))
    for (month, day, year) in datepat.findall(text):
        print('{}-{}-{}'.format(year, month, day))
    for m in datepat.finditer(text):
        print(m.groups())
    datepat = re.compile('(\\d+)/(\\d+)/(\\d+)$')
    print(datepat.match('11/27/2012abcdef'))
    print(datepat.match('11/27/2012'))
    print(re.findall('(\\d+)/(\\d+)/(\\d+)', text))
if __name__ == '__main__':
    match_search()