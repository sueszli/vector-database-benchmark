"""
Topic: 最短匹配，非贪婪模式匹配
Desc : 
"""
import re

def short_match():
    if False:
        print('Hello World!')
    str_pat = re.compile('\\"(.*)\\"')
    text1 = 'Computer says "no."'
    print(str_pat.findall(text1))
    text2 = 'Computer says "no." Phone says "yes."'
    print(str_pat.findall(text2))
    str_pat = re.compile('\\"(.*?)\\"')
    print(str_pat.findall(text2))
if __name__ == '__main__':
    short_match()