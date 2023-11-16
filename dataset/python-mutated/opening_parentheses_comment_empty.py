()
a2 = ()
a3 = f()
a4 = () = a4
a5: List() = 5
raise ()
raise b1b from ()
raise () from b1c
del ()
assert (), ()

def g():
    if False:
        for i in range(10):
            print('nop')
    'Statements that are only allowed in function bodies'
    return ()
    yield ()

async def h():
    """Statements that are only allowed in async function bodies"""
    await ()
with ():
    pass
match ():
    case d2:
        pass
match d3:
    case []:
        pass
while ():
    pass
if ():
    pass
elif ():
    pass
for () in ():
    pass
try:
    pass
except ():
    pass

def e1():
    if False:
        i = 10
        return i + 15
    pass

def e2() -> ():
    if False:
        i = 10
        return i + 15
    pass

class E3:
    pass
f1 = []
[]
f3 = {}
{}