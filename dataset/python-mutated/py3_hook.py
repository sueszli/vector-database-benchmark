import sys

def main():
    if False:
        return 10
    print(sys.version_info[0])
    print(repr(sys.argv[1:]))
    print('Hello World')
    return 0