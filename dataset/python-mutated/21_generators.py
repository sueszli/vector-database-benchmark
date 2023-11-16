def next_square():
    if False:
        for i in range(10):
            print('nop')
    i = 1
    while True:
        yield (i * i)
        i += 1
for n in next_square():
    if n > 25:
        break
    print(n)