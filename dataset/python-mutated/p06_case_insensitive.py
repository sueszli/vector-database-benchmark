"""
Topic: 忽略大小写
Desc : 
"""
import re

def matchcase(word):
    if False:
        return 10

    def replace(m):
        if False:
            for i in range(10):
                print('nop')
        text = m.group()
        if text.isupper():
            return word.upper()
        elif text.islower():
            return word.lower()
        elif text[0].isupper():
            return word.capitalize()
        else:
            return word
    return replace

def case_insens():
    if False:
        print('Hello World!')
    text = 'UPPER PYTHON, lower python, Mixed Python'
    print(re.findall('python', text, flags=re.IGNORECASE))
    print(re.sub('python', 'snake', text, flags=re.IGNORECASE))
    print(re.sub('python', matchcase('snake'), text, flags=re.IGNORECASE))
if __name__ == '__main__':
    case_insens()