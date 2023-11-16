import secrets
from random import random, choice
a = 'Hello'
ok1 = ' '.join([a, ' World'])
ok2 = ''.join(['Finally, ', a, ' World'])
ok3 = 'x'.join(('1', '2', '3'))
ok4 = 'y'.join([1, 2, 3])
ok5 = 'a'.join([random(), random()])
ok6 = 'a'.join([secrets.token_urlsafe(), secrets.token_hex()])
nok1 = 'x'.join({'4', '5', 'yee'})
nok2 = a.join(['1', '2', '3'])
nok3 = 'a'.join(a)
nok4 = 'a'.join([a, a, *a])
nok5 = 'a'.join([choice('flarp')])
nok6 = 'a'.join((x for x in 'feefoofum'))
nok7 = 'a'.join([f'foo{8}', 'bar'])

def create_file_public_url(url, filename):
    if False:
        for i in range(10):
            print('nop')
    return ''.join([url, filename])