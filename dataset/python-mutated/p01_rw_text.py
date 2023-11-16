"""
Topic: 读写文本文件
Desc : 
"""

def rw_text():
    if False:
        i = 10
        return i + 15
    with open('somefile.txt', 'rt') as f:
        for line in f:
            print(line)
    with open('somefile.txt', 'wt') as f:
        f.write('text1')
        f.write('text2')
if __name__ == '__main__':
    rw_text()