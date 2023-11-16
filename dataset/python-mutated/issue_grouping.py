def sink_a(arg):
    if False:
        return 10
    pass

def sink_b(arg):
    if False:
        for i in range(10):
            print('nop')
    pass

def source_a():
    if False:
        i = 10
        return i + 15
    return 1

def source_b():
    if False:
        return 10
    return 2

def multi_sink(d):
    if False:
        while True:
            i = 10
    sink_a(d['a'])
    sink_b(d['b'])

def issue():
    if False:
        while True:
            i = 10
    d = {}
    d['a'] = source_a()
    d['b'] = source_b()
    multi_sink(d)