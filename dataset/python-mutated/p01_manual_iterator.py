"""
Topic: 手动遍历迭代器
Desc : 
"""

def manual_iter():
    if False:
        return 10
    with open('/etc/passwd') as f:
        try:
            while True:
                line = next(f)
                print(line, end='')
        except StopIteration:
            pass

def manual_iter2():
    if False:
        print('Hello World!')
    with open('/etc/passwd') as f:
        while True:
            line = next(f)
            if line is None:
                break
            print(line, end='')
if __name__ == '__main__':
    manual_iter()