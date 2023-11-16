"""
Topic: 单方法类转换为函数
Desc : 
"""
from urllib.request import urlopen

class UrlTemplate:

    def __init__(self, template):
        if False:
            return 10
        self.template = template

    def open(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return urlopen(self.template.format_map(kwargs))
yahoo = UrlTemplate('http://finance.yahoo.com/d/quotes.csv?s={names}&f={fields}')
for line in yahoo.open(names='IBM,AAPL,FB', fields='sl1c1v'):
    print(line.decode('utf-8'))

def urltemplate(template):
    if False:
        for i in range(10):
            print('nop')

    def opener(**kwargs):
        if False:
            return 10
        return urlopen(template.format_map(kwargs))
    return opener
yahoo = urltemplate('http://finance.yahoo.com/d/quotes.csv?s={names}&f={fields}')
for line in yahoo(names='IBM,AAPL,FB', fields='sl1c1v'):
    print(line.decode('utf-8'))