import custom
from custom import ImmutableTypeB

def okay(foo: ImmutableTypeB=[]):
    if False:
        while True:
            i = 10
    ...

def okay(foo: custom.ImmutableTypeA=[]):
    if False:
        for i in range(10):
            print('nop')
    ...

def okay(foo: custom.ImmutableTypeB=[]):
    if False:
        for i in range(10):
            print('nop')
    ...

def error_due_to_missing_import(foo: ImmutableTypeA=[]):
    if False:
        print('Hello World!')
    ...