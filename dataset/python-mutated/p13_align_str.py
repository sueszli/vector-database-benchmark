"""
Topic: 对齐字符串
Desc : 
"""

def align_str():
    if False:
        i = 10
        return i + 15
    text = 'Hello World'
    print(text.ljust(20))
    print(text.rjust(20))
    print(text.center(20))
    print(text.rjust(20, '='))
    print(text.center(20, '*'))
    print(format(text, '>20'))
    print(format(text, '<20'))
    print(format(text, '^20'))
    print(format(text, '=>20s'))
    print(format(text, '*^20s'))
    print('{:=>10s} {:*^10s}'.format('Hello', 'World'))
    x = 1.2345
    print(format(x, '=^10.2f'))
if __name__ == '__main__':
    align_str()