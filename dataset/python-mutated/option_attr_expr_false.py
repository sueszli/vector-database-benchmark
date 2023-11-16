"""bad string"""

@function_that_makes_it_okay('bad string')
def foo():
    if False:
        while True:
            i = 10
    'bad string'

def bar():
    if False:
        for i in range(10):
            print('nop')
    'bad string'
bar = function_that_makes_it_okay('bad string')(bar)