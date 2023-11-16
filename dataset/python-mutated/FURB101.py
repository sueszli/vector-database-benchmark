def foo():
    if False:
        print('Hello World!')
    ...

def bar(x):
    if False:
        for i in range(10):
            print('nop')
    ...
with open('file.txt') as f:
    x = f.read()
with open('file.txt', 'rb') as f:
    x = f.read()
with open('file.txt', mode='rb') as f:
    x = f.read()
with open('file.txt', encoding='utf8') as f:
    x = f.read()
with open('file.txt', errors='ignore') as f:
    x = f.read()
with open('file.txt', errors='ignore', mode='rb') as f:
    x = f.read()
with open('file.txt', mode='r') as f:
    x = f.read()
with open(foo(), 'rb') as f:
    bar('pre')
    bar(f.read())
    bar('post')
    print('Done')
with open('a.txt') as a, open('b.txt', 'rb') as b:
    x = a.read()
    y = b.read()
with foo() as a, open('file.txt') as b, foo() as c:
    bar(a)
    bar(bar(a + b.read()))
    bar(c)
f2 = open('file2.txt')
with open('file.txt') as f:
    x = f2.read()
with open('file.txt') as f:
    x = f.read(100)
with open('file.txt', foo()) as f:
    x = f.read()
with open('file.txt', mode='a+') as f:
    x = f.read()
with open('file.txt', buffering=1) as f:
    x = f.read()
with open('file.txt', newline='\r\n') as f:
    x = f.read()
with open('file.txt', newline='b') as f:
    x = f.read()
with open('file.txt', 'r+') as f:
    x = f.read()
with open('file.txt') as f:
    x = f.read()
    f.seek(0)
    x += f.read(100)
with open(*filename) as f:
    x = f.read()
with open(**kwargs) as f:
    x = f.read()
with open('file.txt', **kwargs) as f:
    x = f.read()
with open('file.txt', mode='r', **kwargs) as f:
    x = f.read()
with open(*filename, mode='r') as f:
    x = f.read()
with open(*filename, file='file.txt', mode='r') as f:
    x = f.read()