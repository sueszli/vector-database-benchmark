"""
Topic: 去除字符串中多余字符
Desc : 
"""
import re

def strip_str():
    if False:
        while True:
            i = 10
    s = ' hello world \n'
    print(s.strip())
    print(s.lstrip())
    print(s.rstrip())
    t = '-----hello====='
    print(t.lstrip('-'))
    print(t.strip('-='))
    s = ' hello     world \n'
    print(s.strip())
    print(s.replace(' ', ''))
    print(re.sub('\\s+', ' ', s))
    with open('filename') as f:
        lines = (line.strip() for line in f)
        for line in lines:
            pass
if __name__ == '__main__':
    strip_str()