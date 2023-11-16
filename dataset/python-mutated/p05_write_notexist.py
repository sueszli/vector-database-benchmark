"""
Topic: 文件不存在时才写入
Desc : 
"""

def write_noexist():
    if False:
        for i in range(10):
            print('nop')
    with open('D:/work/test1.txt', 'wt') as f:
        f.write('BBBBBBBBBBBB')
    with open('D:/work/tt.txt', 'xt') as f:
        f.write('XXXXXXX')
if __name__ == '__main__':
    write_noexist()