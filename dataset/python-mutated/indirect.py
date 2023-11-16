from b2.util import call_jam_function, bjam_signature

def call(*args):
    if False:
        for i in range(10):
            print('nop')
    a1 = args[0]
    name = a1[0]
    a1tail = a1[1:]
    call_jam_function(name, *(a1tail,) + args[1:])