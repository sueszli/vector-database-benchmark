def noop_task():
    if False:
        i = 10
        return i + 15
    print('noop task')
    yield 1

def raise_task():
    if False:
        i = 10
        return i + 15
    print('raise task')
    yield 2
    print('raising')
    raise Exception

def main():
    if False:
        i = 10
        return i + 15
    try:
        yield from raise_task()
    except:
        print('main exception')
    yield from noop_task()
for z in main():
    print('outer iter', z)