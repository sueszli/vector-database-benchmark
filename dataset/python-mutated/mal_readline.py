import os

def readline(prompt):
    if False:
        print('Hello World!')
    res = ''
    os.write(1, prompt)
    while True:
        buf = os.read(0, 255)
        if not buf:
            raise EOFError()
        res += buf
        if res[-1] == '\n':
            return res[:-1]