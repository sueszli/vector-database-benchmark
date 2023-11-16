"""
Topic: 操作路径名
Desc : 
"""
import os

def path_names():
    if False:
        i = 10
        return i + 15
    path = '/Users/beazley/Data/data.csv'
    print(os.path.basename(path))
    print(os.path.dirname(path))
    print(os.path.join('tmp', 'data', os.path.basename(path)))
    path = '~/Data/data.csv'
    print(os.path.expanduser(path))
    print(os.path.splitext(path))
if __name__ == '__main__':
    path_names()