def each_ascii(s):
    if False:
        for i in range(10):
            print('nop')
    for ch in s:
        yield ord(ch)
    return '%s chars' % len(s)

def yield_from(s):
    if False:
        for i in range(10):
            print('nop')
    r = (yield from each_ascii(s))
    print(r)

def main():
    if False:
        while True:
            i = 10
    for x in each_ascii('abc'):
        print(x)
    it = each_ascii('xyz')
    try:
        while True:
            print(next(it))
    except StopIteration as s:
        print(s.value)
    for ch in yield_from('hello'):
        pass
main()