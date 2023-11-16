def trivial():
    if False:
        for i in range(10):
            print('nop')
    pass

def expr_as_statement():
    if False:
        while True:
            i = 10
    61453

def sequential(n):
    if False:
        return 10
    k = n + 4
    s = k + n
    return s

def if_elif_else_dead_path(n):
    if False:
        i = 10
        return i + 15
    if n > 3:
        return 'bigger than three'
    elif n > 4:
        return 'is never executed'
    else:
        return 'smaller than or equal to three'

def nested_ifs():
    if False:
        for i in range(10):
            print('nop')
    if n > 3:
        if n > 4:
            return 'bigger than four'
        else:
            return 'bigger than three'
    else:
        return 'smaller than or equal to three'

def for_loop():
    if False:
        return 10
    for i in range(10):
        print(i)

def for_else(mylist):
    if False:
        while True:
            i = 10
    for i in mylist:
        print(i)
    else:
        print(None)

def recursive(n):
    if False:
        for i in range(10):
            print('nop')
    if n > 4:
        return f(n - 1)
    else:
        return n

def nested_functions():
    if False:
        for i in range(10):
            print('nop')

    def a():
        if False:
            while True:
                i = 10

        def b():
            if False:
                while True:
                    i = 10
            pass
        b()
    a()

def try_else():
    if False:
        i = 10
        return i + 15
    try:
        print(1)
    except TypeA:
        print(2)
    except TypeB:
        print(3)
    else:
        print(4)

def nested_try_finally():
    if False:
        while True:
            i = 10
    try:
        try:
            print(1)
        finally:
            print(2)
    finally:
        print(3)

async def foobar(a, b, c):
    await whatever(a, b, c)
    if await b:
        pass
    async with c:
        pass
    async for x in a:
        pass

def annotated_assign():
    if False:
        for i in range(10):
            print('nop')
    x: Any = None

class Class:

    def handle(self, *args, **options):
        if False:
            return 10
        if args:
            return

        class ServiceProvider:

            def a(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def b(self, data):
                if False:
                    i = 10
                    return i + 15
                if not args:
                    pass

        class Logger:

            def c(*args, **kwargs):
                if False:
                    return 10
                pass

            def error(self, message):
                if False:
                    return 10
                pass

            def info(self, message):
                if False:
                    i = 10
                    return i + 15
                pass

            def exception(self):
                if False:
                    return 10
                pass
        return ServiceProvider(Logger())