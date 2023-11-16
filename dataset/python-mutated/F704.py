def f() -> int:
    if False:
        return 10
    yield 1

class Foo:
    yield 2
yield 3
yield from 3
await f()