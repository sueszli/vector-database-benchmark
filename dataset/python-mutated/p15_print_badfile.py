"""
Topic: 打印不合法文件名
Desc : 
"""
import os
import sys

def bad_filename(filename):
    if False:
        return 10
    return repr(filename)[1:-1]

def bad_filename2(filename):
    if False:
        while True:
            i = 10
    '完美方案'
    temp = filename.encode(sys.getfilesystemencoding(), errors='surrogateescape')
    return temp.decode('latin-1')

def print_badfile(filename):
    if False:
        while True:
            i = 10
    try:
        print(filename)
    except UnicodeEncodeError:
        print('UnicodeEncodeError')
        print(bad_filename(filename))
    files = os.listdir('.')
    print(files)
if __name__ == '__main__':
    print_badfile('bäd.txt')