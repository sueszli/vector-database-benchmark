setup(entry_points={'console_scripts': ['foo-bar=foo.bar.:main']})
run(['ls', '-la'] + path, check=True)

def test_func():
    if False:
        while True:
            i = 10
    if unformatted(args):
        return True
    elif b:
        return True
    return False
if True:
    for _ in range(1):
        print("This won't be formatted")
    print("This won't be formatted either")
else:
    print('This will be formatted')

class A:

    async def call(param):
        if param:
            if param[0:4] in ('ABCD', 'EFGH'):
                print("This won't be formatted")
            elif param[0:4] in ('ZZZZ',):
                print("This won't be formatted either")
        print('This will be formatted')

class Named(t.Protocol):

    @property
    def this_wont_be_formatted(self) -> str:
        if False:
            i = 10
            return i + 15
        ...

class Factory(t.Protocol):

    def this_will_be_formatted(self, **kwargs) -> Named:
        if False:
            for i in range(10):
                print('nop')
        ...
if x:
    return x
elif unformatted:
    will_be_formatted()
setup(entry_points={'console_scripts': ['foo-bar=foo.bar.:main']})
run(['ls', '-la'] + path, check=True)

def test_func():
    if False:
        while True:
            i = 10
    if unformatted(args):
        return True
    elif b:
        return True
    return False
if True:
    for _ in range(1):
        print("This won't be formatted")
    print("This won't be formatted either")
else:
    print('This will be formatted')

class A:

    async def call(param):
        if param:
            if param[0:4] in ('ABCD', 'EFGH'):
                print("This won't be formatted")
            elif param[0:4] in ('ZZZZ',):
                print("This won't be formatted either")
        print('This will be formatted')

class Named(t.Protocol):

    @property
    def this_wont_be_formatted(self) -> str:
        if False:
            while True:
                i = 10
        ...

class Factory(t.Protocol):

    def this_will_be_formatted(self, **kwargs) -> Named:
        if False:
            i = 10
            return i + 15
        ...
if x:
    return x
elif unformatted:
    will_be_formatted()