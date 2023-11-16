def this_is_a_bug():
    if False:
        print('Hello World!')
    o = object()
    if hasattr(o, '__call__'):
        print('Ooh, callable! Or is it?')
    if getattr(o, '__call__', False):
        print('Ooh, callable! Or is it?')

def this_is_fine():
    if False:
        for i in range(10):
            print('nop')
    o = object()
    if callable(o):
        print('Ooh, this is actually callable.')