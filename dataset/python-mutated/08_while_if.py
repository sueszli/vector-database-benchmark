def while_test(a, b, c):
    if False:
        print('Hello World!')
    while a != 2:
        if b:
            a += 1
        elif c:
            c = 0
        else:
            break
    return (a, b, c)

def while1_test(a, b, c):
    if False:
        i = 10
        return i + 15
    while 1:
        if a != 2:
            if b:
                a = 3
                b = 0
            elif c:
                c = 0
            else:
                a += b + c
                break
    return (a, b, c)
assert while_test(2, 0, 0) == (2, 0, 0), 'no while loops'
assert while_test(0, 1, 0) == (2, 1, 0), 'two while loops of b branch'
assert while_test(0, 0, 0) == (0, 0, 0), '0 while loops, else branch'
assert while1_test(4, 1, 1) == (3, 0, 0), 'three while1 loops'
assert while1_test(4, 0, 0) == (4, 0, 0), ' one while1 loop'