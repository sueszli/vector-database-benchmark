def fib():
    if False:
        print('Hello World!')
    (a, b) = (0, 1)
    while True:
        yield a
        (a, b) = (b, a + b)
for f in fib():
    if f > 100:
        break
    print(f)