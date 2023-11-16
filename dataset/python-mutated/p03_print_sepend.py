"""
Topic: print分隔符和结尾符
Desc : 
"""

def print_sepend():
    if False:
        i = 10
        return i + 15
    print('ACME', 50, 91.5)
    print('ACME', 50, 91.5, sep=',')
    print('ACME', 50, 91.5, sep=',', end='!!\n')
    for i in range(5):
        print(i)
    for i in range(5):
        print(i, end=' ')
    print()
    row = ['ACME', 50, 91.5]
    print(*row, sep=',')
if __name__ == '__main__':
    print_sepend()