from integration_test.taint import source, sink

def concatenate_lhs():
    if False:
        print('Hello World!')
    return source() + 'A'

def concatenate_rhs():
    if False:
        i = 10
        return i + 15
    return 'A' + source()

def bad_1():
    if False:
        print('Hello World!')
    a = concatenate_lhs()
    sink(a)

def bad_2():
    if False:
        i = 10
        return i + 15
    a = concatenate_rhs()
    sink(a)