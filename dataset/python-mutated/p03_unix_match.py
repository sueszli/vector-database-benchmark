"""
Topic: Unix Shell通配符匹配
Desc : 
"""
from fnmatch import fnmatch, fnmatchcase

def unix_match():
    if False:
        while True:
            i = 10
    print(fnmatch('foo.txt', '*.txt'))
    print(fnmatch('foo.txt', '?oo.txt'))
    print(fnmatch('Dat45.csv', 'Dat[0-9]*'))
    names = ['Dat1.csv', 'Dat2.csv', 'config.ini', 'foo.py']
    print([name for name in names if fnmatch(name, 'Dat*.csv')])
    print(fnmatchcase('foo.txt', '*.TXT'))
    addresses = ['5412 N CLARK ST', '1060 W ADDISON ST', '1039 W GRANVILLE AVE', '2122 N CLARK ST', '4802 N BROADWAY']
    print([addr for addr in addresses if fnmatchcase(addr, '* ST')])
    print([addr for addr in addresses if fnmatchcase(addr, '54[0-9][0-9] *CLARK*')])
if __name__ == '__main__':
    unix_match()