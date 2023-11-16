"""
Topic: 正则式分割字符串
Desc : 
"""
import re

def split_str():
    if False:
        return 10
    line = 'asdf fjdk; afed, fjek,asdf, foo'
    print(re.split('[;,\\s]\\s*', line))
    print(re.split('(;|,|\\s)\\s*', line))
    fields = re.split('(;|,|\\s)\\s*', line)
    values = fields[::2]
    delimiters = fields[1::2] + ['']
    print(''.join((v + d for (v, d) in zip(values, delimiters))))
if __name__ == '__main__':
    split_str()