from viztracer import VizCounter, VizObject, VizTracer

def h(a):
    if False:
        print('Hello World!')
    counter.a = a
    ob.b = 3
    return 1 / (a - 3)

def g(a, b):
    if False:
        for i in range(10):
            print('nop')
    a += h(a)
    b += 3

def f(a, b):
    if False:
        while True:
            i = 10
    a = a + 2
    ob.s = str(b)
    g(a + 1, b * 2)
    h(36)

def t(a):
    if False:
        while True:
            i = 10
    f(a + 1, a + 2)
    a += 3
    f(a + 5, 2)
tracer = VizTracer()
counter = VizCounter(tracer, 'a')
ob = VizObject(tracer, 'b')
tracer.start()
t(3)
tracer.stop()
tracer.save('vdb_basic.json')