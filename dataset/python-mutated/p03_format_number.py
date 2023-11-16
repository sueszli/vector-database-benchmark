"""
Topic: 格式化输出数字
Desc : 
"""

def format_number():
    if False:
        while True:
            i = 10
    x = 1234.56789
    print(format(x, '0.2f'))
    print(format(x, '>10.1f'))
    print(format(x, '<10.1f'))
    print(format(x, '^10.1f'))
    print(format(x, ','))
    print(format(x, '0,.1f'))
    print(format(x, 'e'))
    print(format(x, '0.2E'))
    print('The value is {:0,.2f}'.format(x))
    print(format(x, '0.1f'))
    print(format(-x, '0.1f'))
    swap_separators = {ord('.'): ',', ord(','): '.'}
    print(format(x, ',').translate(swap_separators))
if __name__ == '__main__':
    format_number()