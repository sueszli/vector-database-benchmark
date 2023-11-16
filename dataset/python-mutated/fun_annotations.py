def foo(x: int, y: list) -> dict:
    if False:
        return 10
    return {x: y}
print(foo(1, [2, 3]))