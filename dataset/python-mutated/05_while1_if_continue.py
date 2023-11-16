def readline(b):
    if False:
        print('Hello World!')
    a = 1
    while True:
        if b:
            if b[0]:
                a = 2
                b = None
                continue
        b = None
        a = 5
        return a
assert readline(None) == 1
assert readline([2]) == 2

def readline2(self):
    if False:
        print('Hello World!')
    while True:
        line = 5
        if self[0]:
            if self:
                self[0] = 1
                continue
        return line + self[0]

def PipeClient(address):
    if False:
        i = 10
        return i + 15
    while 1:
        try:
            address += 1
        except OSError as e:
            raise e
    else:
        raise