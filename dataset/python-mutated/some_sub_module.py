import sys
print('This must be Python3 which no longer needs __init__.py to accept a package.')
print('The parent path is', sys.modules['some_package.sub_package'].__path__)

def s():
    if False:
        for i in range(10):
            print('nop')
    pass
print(s)