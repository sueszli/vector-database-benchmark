try:
    memoryview(b'a').cast
except:
    print('SKIP')
    raise SystemExit
b = bytearray(range(16))

def print_memview(mv):
    if False:
        for i in range(10):
            print('nop')
    print(', '.join((hex(v) for v in mv)))
mv = memoryview(b)
print_memview(mv)
print_memview(mv[4:])
words = mv.cast('I')
print_memview(words)
print_memview(mv[4:].cast('I'))
print_memview(words[1:])
print_memview(words.cast('B'))