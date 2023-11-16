"""
Topic: 在正则式中使用Unicode
Desc : 
"""
import re

def re_unicode():
    if False:
        return 10
    num = re.compile('\\d+')
    print(num.match('123'))
    print(num.match('١٢٣'))
    arabic = re.compile('[\u0600-ۿݐ-ݿࢠ-ࣿ]+')
    pat = re.compile('straße', re.IGNORECASE)
    s = 'straße'
    print(pat.match(s))
    print(pat.match(s.upper()))
    print(s.upper())
if __name__ == '__main__':
    re_unicode()