import sys

def main():
    if False:
        for i in range(10):
            print('nop')
    for line in sys.stdin:
        print(line.replace(' ', '').replace('|', ' ').strip())
if __name__ == '__main__':
    main()