"""
Topic: 输出重定向到文件
Desc : 
"""

def print_tofile():
    if False:
        while True:
            i = 10
    with open('d:/work/test.txt', 'wt') as f:
        print('Hello World!', file=f)
if __name__ == '__main__':
    print_tofile()