"""
Topic: 字符串的I/O操作
Desc : 
"""
import io

def string_io():
    if False:
        return 10
    s = io.StringIO()
    s.write('Hello World\n')
    print('This is a test', file=s)
    print(s.getvalue())
    s = io.StringIO('Hello\nWorld\n')
    print(s.read(4))
    print(s.read())
if __name__ == '__main__':
    string_io()