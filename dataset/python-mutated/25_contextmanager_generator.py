from contextlib import contextmanager

@contextmanager
def open_file(name):
    if False:
        for i in range(10):
            print('nop')
    f = open(name, 'w')
    yield f
    f.close()

def main():
    if False:
        while True:
            i = 10
    with open_file('some_file') as f:
        f.write('Hola!')